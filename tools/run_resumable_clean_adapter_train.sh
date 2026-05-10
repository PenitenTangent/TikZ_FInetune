#!/usr/bin/env bash
# Resumable full-dataset SFT using strict coverage (see configs/clean_adapter_recovery.yaml).
# - State: runs/${RUN_ID}/coverage_state_clean_adapter.json (cursor + fingerprints)
# - Epoch orders: runs/${RUN_ID}/epoch_orders_clean_adapter/
# - Reuse the SAME --run-id and SAME total --iters on every resume; pass --resume-adapter to latest checkpoint.
#
# Stale lock after kill: only if no trainer is running, remove:
#   runs/${RUN_ID}/train_clean_adapter.lock
#
# Usage:
#   FIRST:  bash tools/run_resumable_clean_adapter_train.sh
#   RESUME: RESUME_ADAPTER=runs/.../NNNNNNN_adapters.safetensors bash tools/run_resumable_clean_adapter_train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/clean_adapter_recovery.yaml}"
RUN_ID="${RUN_ID:-final_full_202468}"
TOTAL_ITERS="${TOTAL_ITERS:-202468}"
TRAIN_DATASET="${TRAIN_DATASET:-data/prepared/train_unified.jsonl}"
VAL_DATASET="${VAL_DATASET:-data/prepared/val_small_200.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/${RUN_ID}/clean_adapter.safetensors}"
# Optional; set when resuming from a saved adapter/checkpoint
RESUME_ADAPTER="${RESUME_ADAPTER:-}"

CMD=(
  "$PYTHON_BIN" -u -m tikz_mlx.cli train
  --config "$CONFIG"
  --dataset "$TRAIN_DATASET"
  --val-dataset "$VAL_DATASET"
  --iters "$TOTAL_ITERS"
  --output-path "$OUTPUT_PATH"
  --run-id "$RUN_ID"
  --skip-post-ab-eval
)

if [[ -n "$RESUME_ADAPTER" ]]; then
  CMD+=(--resume-adapter "$RESUME_ADAPTER")
fi

echo "run_id=$RUN_ID total_iters=$TOTAL_ITERS (always pass full budget on resume)"
echo "output=$OUTPUT_PATH"
if [[ -n "$RESUME_ADAPTER" ]]; then
  echo "resume_adapter=$RESUME_ADAPTER"
else
  echo "resume_adapter=(none, fresh start)"
fi
"${CMD[@]}"
