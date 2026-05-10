#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/lora_prod.yaml}"
TRAIN_DATASET="${TRAIN_DATASET:-data/prepared/train_unified.jsonl}"
VAL_DATASET="${VAL_DATASET:-data/prepared/val_unified.jsonl}"
STAGE2_DATASET="${STAGE2_DATASET:-data/prepared/train_stage2.jsonl}"

# Full-run launcher defaults.
STAGE2_ITERS="${STAGE2_ITERS:-200}"
STAGE2_STEPS_PER_SAVE="${STAGE2_STEPS_PER_SAVE:-10}"
STAGE2_STEPS_PER_REPORT="${STAGE2_STEPS_PER_REPORT:-5}"

# Disk safety guard to avoid mid-run checkpoint write failures.
MIN_FREE_GB="${MIN_FREE_GB:-40}"
SKIP_DISK_CHECK="${SKIP_DISK_CHECK:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-full_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-runs/logs/${RUN_TAG}}"
STAGE1_OUT="${STAGE1_OUT:-runs/tikz_lora_adapter_${RUN_TAG}.safetensors}"
STAGE2_OUT="${STAGE2_OUT:-runs/tikz_stage2_adapter_${RUN_TAG}.safetensors}"
STAGE2_OUT_BASENAME="$(basename "$STAGE2_OUT")"
STAGE2_RUN_ID_DEFAULT="${STAGE2_OUT_BASENAME%.safetensors}"
STAGE2_RUN_ID="${STAGE2_RUN_ID:-$STAGE2_RUN_ID_DEFAULT}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN"
    exit 1
fi

require_existing_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        echo "Missing ${label}: $path"
        exit 1
    fi
}

require_nonempty_file() {
    local path="$1"
    local label="$2"
    if ! awk 'NF {found=1; exit} END {exit found?0:1}' "$path"; then
        echo "${label} is empty: $path"
        exit 1
    fi
}

require_existing_file "$CONFIG_PATH" "config file"
require_existing_file "$TRAIN_DATASET" "train dataset"
require_existing_file "$VAL_DATASET" "validation dataset"
require_existing_file "$STAGE2_DATASET" "stage2 dataset"
require_nonempty_file "$TRAIN_DATASET" "Train dataset"
require_nonempty_file "$VAL_DATASET" "Validation dataset"
require_nonempty_file "$STAGE2_DATASET" "Stage2 dataset"

if [[ "$SKIP_DISK_CHECK" != "1" ]]; then
    AVAILABLE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
    AVAILABLE_GB="$(( AVAILABLE_KB / 1024 / 1024 ))"
    if (( AVAILABLE_GB < MIN_FREE_GB )); then
        echo "Insufficient free disk space for full finetune: ${AVAILABLE_GB}GiB available, ${MIN_FREE_GB}GiB required."
        echo "Free disk space or override with MIN_FREE_GB=<lower> (or SKIP_DISK_CHECK=1 if you accept risk)."
        exit 1
    fi
fi

FULL_CONFIG_PATH="$LOG_DIR/config_full.yaml"
STAGE1_LOG="$LOG_DIR/stage1.log"
STAGE2_LOG="$LOG_DIR/stage2.log"
SUMMARY_JSON="$LOG_DIR/run_summary.json"
ADAPTER_DIR="$LOG_DIR/stage1_adapter_dir"
TELEMETRY_COPY="$LOG_DIR/stage2_metrics.jsonl"

# Keep generated config directly under configs/ so relative dataset paths
# resolve exactly like the production config.
FULL_CONFIG_PATH="$ROOT_DIR/configs/full_${RUN_TAG}.yaml"

"$PYTHON_BIN" - "$CONFIG_PATH" "$FULL_CONFIG_PATH" "$STAGE2_ITERS" "$STAGE2_STEPS_PER_SAVE" "$STAGE2_STEPS_PER_REPORT" <<'PY'
import pathlib
import sys

import yaml

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
stage2_iters = int(sys.argv[3])
stage2_steps_per_save = int(sys.argv[4])
stage2_steps_per_report = int(sys.argv[5])

if stage2_iters <= 1:
    raise ValueError(f"STAGE2_ITERS must be > 1 for full finetune, got {stage2_iters}")
if stage2_steps_per_save <= 0:
    raise ValueError(
        f"STAGE2_STEPS_PER_SAVE must be > 0 for full finetune, got {stage2_steps_per_save}"
    )
if stage2_steps_per_report <= 0:
    raise ValueError(
        f"STAGE2_STEPS_PER_REPORT must be > 0 for full finetune, got {stage2_steps_per_report}"
    )

data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
training = data.setdefault("training", {})
training["allow_full_training"] = True
stage2 = training.setdefault("stage2", {})
stage2["allow_full_training"] = True
stage2["enabled"] = True
stage2["iters"] = stage2_iters
stage2["steps_per_save"] = stage2_steps_per_save
stage2["steps_per_report"] = stage2_steps_per_report
target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

echo "Full-finetune config written to $FULL_CONFIG_PATH"
echo "Stage2 full settings: iters=$STAGE2_ITERS steps_per_save=$STAGE2_STEPS_PER_SAVE steps_per_report=$STAGE2_STEPS_PER_REPORT"

stage1_cmd=(
    "$PYTHON_BIN" -u -m tikz_mlx.cli train
    --config "$FULL_CONFIG_PATH"
    --dataset "$TRAIN_DATASET"
    --val-dataset "$VAL_DATASET"
    --output-path "$STAGE1_OUT"
)

echo "Running stage1 full finetune with train and validation datasets"
echo "Stage1 log: $STAGE1_LOG"
"${stage1_cmd[@]}" 2>&1 | tee "$STAGE1_LOG"

if [[ ! -f "runs/adapter_config.json" ]]; then
    echo "Missing runs/adapter_config.json after stage1 run"
    exit 1
fi
if [[ ! -f "$STAGE1_OUT" ]]; then
    echo "Missing stage1 adapter output: $STAGE1_OUT"
    exit 1
fi

mkdir -p "$ADAPTER_DIR"
cp "runs/adapter_config.json" "$ADAPTER_DIR/adapter_config.json"
cp "$STAGE1_OUT" "$ADAPTER_DIR/adapters.safetensors"

stage2_cmd=(
    "$PYTHON_BIN" -u -m tikz_mlx.cli train-stage2
    --config "$FULL_CONFIG_PATH"
    --dataset "$STAGE2_DATASET"
    --output-path "$STAGE2_OUT"
    --resume-adapter "$ADAPTER_DIR"
    --run-id "$STAGE2_RUN_ID"
)

echo "Running stage2 full finetune"
echo "Stage2 log: $STAGE2_LOG"
"${stage2_cmd[@]}" 2>&1 | tee "$STAGE2_LOG"

if [[ ! -f "$STAGE2_OUT" ]]; then
    echo "Missing stage2 adapter output: $STAGE2_OUT"
    exit 1
fi

RUN_METADATA_PATH="runs/${STAGE2_RUN_ID}/run_metadata.json"
TELEMETRY_SOURCE=""
if [[ -f "$RUN_METADATA_PATH" ]]; then
    TELEMETRY_SOURCE="$("$PYTHON_BIN" - "$RUN_METADATA_PATH" <<'PY'
import json
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    print("")
    raise SystemExit(0)

value = payload.get("telemetry_path")
if isinstance(value, str):
    print(value)
else:
    print("")
PY
)"
fi

if [[ -n "$TELEMETRY_SOURCE" && -f "$TELEMETRY_SOURCE" ]]; then
    cp "$TELEMETRY_SOURCE" "$TELEMETRY_COPY"
elif [[ -f "runs/stage2_checkpoints/metrics.jsonl" ]]; then
    cp "runs/stage2_checkpoints/metrics.jsonl" "$TELEMETRY_COPY"
fi

"$PYTHON_BIN" - "$SUMMARY_JSON" "$FULL_CONFIG_PATH" "$STAGE1_LOG" "$STAGE2_LOG" "$STAGE1_OUT" "$STAGE2_OUT" "$ADAPTER_DIR" "$TELEMETRY_COPY" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
payload = {
    "config_path": sys.argv[2],
    "stage1_log": sys.argv[3],
    "stage2_log": sys.argv[4],
    "stage1_output": sys.argv[5],
    "stage2_output": sys.argv[6],
    "stage2_resume_adapter_dir": sys.argv[7],
    "telemetry_copy": sys.argv[8],
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "Full finetune run complete"
echo "Summary file: $SUMMARY_JSON"