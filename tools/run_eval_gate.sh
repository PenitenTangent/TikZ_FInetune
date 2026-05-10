#!/bin/bash
set -e

# Usage: ./tools/run_eval_gate.sh --config <config> --adapter <adapter_path> [--num-samples 32]

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

EVAL_DIR="outputs/eval_gate_$(date +%Y%m%d_%H%M%S)"
echo "Starting post-training sentinel eval in $EVAL_DIR..."

python3 tools/ab_eval.py \
  --config "$CONFIG" \
  --adapter-path "$ADAPTER" \
  --num-samples "$NUM_SAMPLES" \
  --seed 42 \
  --max-tokens 2048 \
  --out-dir "$EVAL_DIR"

echo "Checking promotion gate thresholds..."
if python3 tools/check_promotion_gate.py --eval-dir "$EVAL_DIR"; then
    echo "✅ Gate PASSED."
    exit 0
else
    echo "❌ Gate FAILED. Quarantining adapter..."
    # Add to manifest
    python3 tools/quarantine_adapters.py
    exit 1
fi
