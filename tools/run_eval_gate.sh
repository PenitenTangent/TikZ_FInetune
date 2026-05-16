#!/bin/bash
# Post-training evaluation gate.
# Runs a sentinel A/B eval and enforces hard quality thresholds.
# On failure, quarantines the specific adapter and writes a failure bundle.
#
# Usage:
#   ./tools/run_eval_gate.sh --config <config> --adapter <adapter_path> --stage <stage0|...|stage5|normal> [--num-samples 100]
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG=""
ADAPTER=""
STAGE=""
GATE_CONFIG="configs/promotion_gate.yaml"
NUM_SAMPLES=100
SENTINEL_MANIFEST="data/manifests/sentinel_100_deleaked.json"
EVAL_DIR=""
BASE_CACHE_DIR="${BASE_EVAL_CACHE_DIR:-outputs/eval_base_cache}"
USE_BASE_CACHE=1
COMPILE_WORKERS="${EVAL_COMPILE_WORKERS:-4}"
NO_QUARANTINE=0
SKIP_COLLAPSE_PROBE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --adapter) ADAPTER="$2"; shift ;;
        --stage) STAGE="$2"; shift ;;
        --gate-config) GATE_CONFIG="$2"; shift ;;
        --num-samples) NUM_SAMPLES="$2"; shift ;;
        --sentinel-manifest) SENTINEL_MANIFEST="$2"; shift ;;
        --out-dir) EVAL_DIR="$2"; shift ;;
        --base-cache-dir) BASE_CACHE_DIR="$2"; shift ;;
        --no-base-cache) USE_BASE_CACHE=0 ;;
        --compile-workers) COMPILE_WORKERS="$2"; shift ;;
        --no-quarantine) NO_QUARANTINE=1 ;;
        --skip-collapse-probe) SKIP_COLLAPSE_PROBE=1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$CONFIG" ] || [ -z "$ADAPTER" ] || [ -z "$STAGE" ]; then
    echo "Usage: $0 --config <config> --adapter <adapter_path> --stage <stage0|...|stage5|normal> [--num-samples 100] [--sentinel-manifest <path>]"
    exit 1
fi

cd "$PROJECT_ROOT"

if [ -z "$EVAL_DIR" ]; then
    adapter_slug="$(basename "$ADAPTER" | tr -c '[:alnum:]_.-' '_')"
    EVAL_DIR="outputs/eval_gate_$(date +%Y%m%d_%H%M%S)_$$_${adapter_slug}"
fi
mkdir -p "$EVAL_DIR"
echo ""
echo "========================================="
echo "  Post-Training Sentinel Eval"
echo "  Adapter: $ADAPTER"
echo "  Output:  $EVAL_DIR"
echo "========================================="

# 1. Run A/B eval (base + finetuned)
ab_eval_args=(
  --config "$CONFIG"
  --adapter-path "$ADAPTER"
  --sentinel-manifest "$SENTINEL_MANIFEST"
  --num-samples "$NUM_SAMPLES"
  --max-tokens 2048
  --out-dir "$EVAL_DIR"
  --collapse-probe-out "$EVAL_DIR/collapse_probe.json"
  --compile-workers "$COMPILE_WORKERS"
)
if [ "$USE_BASE_CACHE" -eq 1 ]; then
  ab_eval_args+=(--base-cache-dir "$BASE_CACHE_DIR")
else
  ab_eval_args+=(--no-base-cache)
fi
if [ "$SKIP_COLLAPSE_PROBE" -eq 1 ]; then
  ab_eval_args+=(--skip-collapse-probe)
fi

AB_EVAL_PASSED=0
set +e
"$PYTHON_EXE" tools/ab_eval.py \
  "${ab_eval_args[@]}"
ab_eval_status=$?
set -e
if [ "$ab_eval_status" -eq 0 ]; then
    AB_EVAL_PASSED=1
else
    echo "ERROR: A/B eval or collapse probe failed with exit code $ab_eval_status."
fi

# 2. Check promotion gate
echo ""
echo "Checking promotion gate thresholds..."
GATE_PASSED=0
if [ "$AB_EVAL_PASSED" -eq 1 ] && "$PYTHON_EXE" tools/check_promotion_gate.py \
    --eval-dir "$EVAL_DIR" \
    --stage "$STAGE" \
    --gate-config "$GATE_CONFIG"; then
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
    cp "$EVAL_DIR/worst_cases.json" "$BUNDLE_DIR/worst_cases.json" 2>/dev/null || true
    cp "$EVAL_DIR/collapse_probe.json" "$BUNDLE_DIR/collapse_probe.json" 2>/dev/null || true
    
    echo "❌ Gate FAILED. Failure bundle written to: $BUNDLE_DIR"
    
    # Quarantine the specific adapter
    if [ "$NO_QUARANTINE" -eq 0 ]; then
        echo "Quarantining adapter: $ADAPTER"
        "$PYTHON_EXE" tools/quarantine_adapters.py \
            --adapter "$ADAPTER" \
            --reason "post-training eval gate failed (sentinel-${NUM_SAMPLES})"
    else
        echo "No-quarantine mode enabled; leaving adapter unquarantined."
    fi
    
    exit 1
fi

echo "✅ Gate PASSED."
exit 0
