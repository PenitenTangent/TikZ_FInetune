#!/bin/bash
# Run all post-stage gates for a single adapter:
# gradient telemetry, collapse probe, sentinel A/B eval, and promotion gate.
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG=""
ADAPTER=""
CHECKPOINT_DIR=""
NUM_SAMPLES=100
NUM_SAMPLES_SET=0
SENTINEL_MANIFEST="data/manifests/sentinel_100_deleaked.json"
ALLOW_MISSING_GRADIENT_TELEMETRY=0
SKIP_GRADIENT_TELEMETRY=0
SKIP_COLLAPSE_PROBE=0
SKIP_EVAL_GATE=0
OUT_DIR=""
GATE_MODE="promote"
BASE_CACHE_DIR="${BASE_EVAL_CACHE_DIR:-outputs/eval_base_cache}"
COMPILE_WORKERS="${EVAL_COMPILE_WORKERS:-4}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; NUM_SAMPLES_SET=1; shift 2 ;;
    --sentinel-manifest) SENTINEL_MANIFEST="$2"; shift 2 ;;
    --gate-mode) GATE_MODE="$2"; shift 2 ;;
    --base-cache-dir) BASE_CACHE_DIR="$2"; shift 2 ;;
    --compile-workers) COMPILE_WORKERS="$2"; shift 2 ;;
    --allow-missing-gradient-telemetry) ALLOW_MISSING_GRADIENT_TELEMETRY=1; shift ;;
    --skip-gradient-telemetry) SKIP_GRADIENT_TELEMETRY=1; shift ;;
    --skip-collapse-probe) SKIP_COLLAPSE_PROBE=1; shift ;;
    --skip-eval-gate) SKIP_EVAL_GATE=1; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

if [ -z "$CONFIG" ] || [ -z "$ADAPTER" ]; then
  echo "Usage: $0 --config <config> --adapter <adapter> [--checkpoint-dir <dir>] [--num-samples 100]"
  exit 1
fi
case "$GATE_MODE" in
  quick|full|promote) ;;
  *) echo "ERROR: --gate-mode must be quick, full, or promote."; exit 1 ;;
esac
if [ "$NUM_SAMPLES_SET" -eq 0 ] && [ "$GATE_MODE" = "quick" ]; then
  NUM_SAMPLES=32
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "$CHECKPOINT_DIR" ]; then
  CHECKPOINT_DIR="$(dirname "$ADAPTER")"
fi
if [ -z "$OUT_DIR" ]; then
  adapter_slug="$(basename "$ADAPTER" | tr -c '[:alnum:]_.-' '_')"
  OUT_DIR="outputs/stage_gate_$(date +%Y%m%d_%H%M%S)_${adapter_slug}"
fi
mkdir -p "$OUT_DIR"

echo ""
echo "========================================="
echo "  Stage Gate"
echo "  Adapter:    $ADAPTER"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Output:     $OUT_DIR"
echo "  Mode:       $GATE_MODE"
echo "========================================="

if [ "$SKIP_GRADIENT_TELEMETRY" -eq 0 ]; then
  TELEMETRY_PATH="$CHECKPOINT_DIR/gradient_clip_telemetry.jsonl"
  LEGACY_TELEMETRY_PATH="$(dirname "$ADAPTER")/gradient_clip_telemetry.jsonl"
  if [ ! -f "$TELEMETRY_PATH" ] && [ -f "$LEGACY_TELEMETRY_PATH" ]; then
    echo "WARNING: using legacy adapter-dir gradient telemetry: $LEGACY_TELEMETRY_PATH"
    TELEMETRY_PATH="$LEGACY_TELEMETRY_PATH"
  fi
  telemetry_args=(--telemetry "$TELEMETRY_PATH" --out "$OUT_DIR/gradient_telemetry_gate.json")
  if [ "$ALLOW_MISSING_GRADIENT_TELEMETRY" -eq 1 ]; then
    telemetry_args+=(--allow-missing)
  fi
  "$PYTHON_EXE" tools/check_gradient_telemetry.py "${telemetry_args[@]}"
fi

if [ "$SKIP_COLLAPSE_PROBE" -eq 0 ] && [ "$SKIP_EVAL_GATE" -eq 1 ]; then
  "$PYTHON_EXE" tools/run_collapse_probe.py \
    --config "$CONFIG" \
    --adapter "$ADAPTER" \
    --out "$OUT_DIR/collapse_probe.json"
fi

if [ "$SKIP_EVAL_GATE" -eq 0 ]; then
  eval_args=(
    --config "$CONFIG"
    --adapter "$ADAPTER"
    --num-samples "$NUM_SAMPLES"
    --sentinel-manifest "$SENTINEL_MANIFEST"
    --out-dir "$OUT_DIR/eval"
    --base-cache-dir "$BASE_CACHE_DIR"
    --compile-workers "$COMPILE_WORKERS"
  )
  if [ "$SKIP_COLLAPSE_PROBE" -eq 1 ]; then
    eval_args+=(--skip-collapse-probe)
  fi
  if [ "$GATE_MODE" != "promote" ]; then
    eval_args+=(--no-quarantine)
  fi
  bash tools/run_eval_gate.sh \
    "${eval_args[@]}"
fi

cat > "$OUT_DIR/stage_gate_result.json" <<JSON
{
  "passed": true,
  "adapter": "$ADAPTER",
  "checkpoint_dir": "$CHECKPOINT_DIR",
  "config": "$CONFIG",
  "gate_mode": "$GATE_MODE",
  "num_samples": $NUM_SAMPLES
}
JSON

echo "Stage gate passed: $ADAPTER"
