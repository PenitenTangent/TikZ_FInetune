#!/bin/bash
# Post-training evaluation gate.
# Runs a sentinel A/B eval and enforces hard quality thresholds.
# On failure, quarantines the specific adapter and writes a failure bundle.
#
# Usage:
#   ./tools/run_eval_gate.sh --config <config> --adapter <adapter_path> [--num-samples 32]
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG=""
ADAPTER=""
NUM_SAMPLES=32

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --adapter) ADAPTER="$2"; shift ;;
        --num-samples) NUM_SAMPLES="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$CONFIG" ] || [ -z "$ADAPTER" ]; then
    echo "Usage: $0 --config <config> --adapter <adapter_path> [--num-samples 32]"
    exit 1
fi

cd "$PROJECT_ROOT"

EVAL_DIR="outputs/eval_gate_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "========================================="
echo "  Post-Training Sentinel Eval"
echo "  Adapter: $ADAPTER"
echo "  Output:  $EVAL_DIR"
echo "========================================="

# 1. Run A/B eval (base + finetuned)
"$PYTHON_EXE" tools/ab_eval.py \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --num-samples "$NUM_SAMPLES" \
  --seed 42 \
  --max-tokens 2048 \
  --out-dir "$EVAL_DIR"

# 2. Check promotion gate
echo ""
echo "Checking promotion gate thresholds..."
GATE_PASSED=0
if "$PYTHON_EXE" tools/check_promotion_gate.py --eval-dir "$EVAL_DIR"; then
    GATE_PASSED=1
fi

# 3. Write failure bundle if gate failed
if [ "$GATE_PASSED" -eq 0 ]; then
    BUNDLE_DIR="$EVAL_DIR/failure_bundle"
    mkdir -p "$BUNDLE_DIR"
    
    # Collect adapter SHA256
    shasum -a 256 "$ADAPTER" | awk '{print $1}' > "$BUNDLE_DIR/adapter_sha256.txt" 2>/dev/null || true
    
    # Copy config
    cp "$CONFIG" "$BUNDLE_DIR/config.yaml" 2>/dev/null || true
    
    # Git SHA
    git rev-parse HEAD > "$BUNDLE_DIR/git_sha.txt" 2>/dev/null || echo "unknown" > "$BUNDLE_DIR/git_sha.txt"
    
    # Copy gate result
    cp "$EVAL_DIR/promotion_gate_result.json" "$BUNDLE_DIR/metrics.json" 2>/dev/null || true
    cp "$EVAL_DIR/results.json" "$BUNDLE_DIR/results.json" 2>/dev/null || true
    
    echo "❌ Gate FAILED. Failure bundle written to: $BUNDLE_DIR"
    
    # Quarantine the specific adapter
    echo "Quarantining adapter: $ADAPTER"
    "$PYTHON_EXE" tools/quarantine_adapters.py \
        --adapter "$ADAPTER" \
        --reason "post-training eval gate failed (sentinel-${NUM_SAMPLES})"
    
    exit 1
fi

echo "✅ Gate PASSED."
exit 0
