#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/lora_prod.yaml}"
TRAIN_DATASET="${TRAIN_DATASET:-data/prepared/train_unified.jsonl}"
VAL_DATASET="${VAL_DATASET:-data/prepared/val_unified.jsonl}"
GOLD_EVAL_DATASET="${GOLD_EVAL_DATASET:-data/prepared/gold_eval_unified.jsonl}"

# Phase1 launch defaults.
AB_SAMPLE_SIZE="${AB_SAMPLE_SIZE:-120}"
AB_SEED="${AB_SEED:-20260419}"
AB_MAX_TOKENS_CAP="${AB_MAX_TOKENS_CAP:-2048}"
AB_HYBRID_VISUAL_THRESHOLD="${AB_HYBRID_VISUAL_THRESHOLD:-0.75}"
REWARD_BACKEND="${REWARD_BACKEND:-emd}"
STAGE1_ITERS="${STAGE1_ITERS:-}"

# Strict Stage1 quality gate defaults.
MIN_SUBSTANTIVE_COMPILE_DELTA="${MIN_SUBSTANTIVE_COMPILE_DELTA:-0.0}"
MIN_SUBSTANTIVE_TIKZ_DELTA="${MIN_SUBSTANTIVE_TIKZ_DELTA:-0.0}"
MIN_STAGE1_SUBSTANTIVE_COMPILE_RATE="${MIN_STAGE1_SUBSTANTIVE_COMPILE_RATE:-0.20}"
MIN_STAGE1_SUBSTANTIVE_TIKZ_RATE="${MIN_STAGE1_SUBSTANTIVE_TIKZ_RATE:-0.75}"
GATE_CONFIG_PATH="${GATE_CONFIG_PATH:-configs/gate_config_v1.json}"

# Operational controls.
MIN_FREE_GB="${MIN_FREE_GB:-40}"
SKIP_DISK_CHECK="${SKIP_DISK_CHECK:-0}"
STAGE1_DRY_RUN="${STAGE1_DRY_RUN:-0}"
PROMOTE_ON_PASS="${PROMOTE_ON_PASS:-1}"
FORCE_POLICY_INIT="${FORCE_POLICY_INIT:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-phase1_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-runs/logs/${RUN_TAG}}"
STAGE1_OUT="${STAGE1_OUT:-runs/tikz_lora_adapter_${RUN_TAG}.safetensors}"
STAGE1_OUT_BASENAME="$(basename "$STAGE1_OUT")"
STAGE1_RUN_ID_DEFAULT="${STAGE1_OUT_BASENAME%.safetensors}"
STAGE1_RUN_ID="${STAGE1_RUN_ID:-$STAGE1_RUN_ID_DEFAULT}"

PHASE1_CONFIG_PATH="$ROOT_DIR/configs/phase1_${RUN_TAG}.yaml"
STAGE1_LOG="$LOG_DIR/stage1.log"
AB_EVAL_OUT_DIR="$LOG_DIR/stage1_ab_eval_strict"
AB_REPORT_PATH="$AB_EVAL_OUT_DIR/report.json"
PROMOTION_RESULT_JSON="$LOG_DIR/stage1_promotion_result.json"
SUMMARY_JSON="$LOG_DIR/phase1_summary.json"
ADAPTER_DIR="$LOG_DIR/stage1_adapter_dir"
SFT_FINAL_PATH="${SFT_FINAL_PATH:-runs/sft_final.safetensors}"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-runs/policy_init.safetensors}"

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
require_existing_file "$GOLD_EVAL_DATASET" "gold eval dataset"
require_nonempty_file "$TRAIN_DATASET" "Train dataset"
require_nonempty_file "$VAL_DATASET" "Validation dataset"
require_nonempty_file "$GOLD_EVAL_DATASET" "Gold eval dataset"

if [[ "$SKIP_DISK_CHECK" != "1" ]]; then
    AVAILABLE_KB="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
    AVAILABLE_GB="$(( AVAILABLE_KB / 1024 / 1024 ))"
    if (( AVAILABLE_GB < MIN_FREE_GB )); then
        echo "Insufficient free disk space for phase1 finetune: ${AVAILABLE_GB}GiB available, ${MIN_FREE_GB}GiB required."
        echo "Free disk space or override with MIN_FREE_GB=<lower> (or SKIP_DISK_CHECK=1 if you accept risk)."
        exit 1
    fi
fi

"$PYTHON_BIN" - "$CONFIG_PATH" "$PHASE1_CONFIG_PATH" <<'PY'
import pathlib
import sys

import yaml

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])

data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
training = data.setdefault("training", {})
training["allow_full_training"] = True
stage2 = training.setdefault("stage2", {})
stage2["enabled"] = False
stage2["allow_full_training"] = False

target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

echo "Phase1 config written to $PHASE1_CONFIG_PATH"
echo "Stage2 disabled for this run"

stage1_cmd=(
    "$PYTHON_BIN" -u -m tikz_mlx.cli train
    --config "$PHASE1_CONFIG_PATH"
    --dataset "$TRAIN_DATASET"
    --val-dataset "$VAL_DATASET"
    --output-path "$STAGE1_OUT"
    --run-id "$STAGE1_RUN_ID"
)
if [[ -n "$STAGE1_ITERS" ]]; then
    stage1_cmd+=(--iters "$STAGE1_ITERS")
fi

if [[ "$STAGE1_DRY_RUN" == "1" ]]; then
    stage1_cmd+=(--dry-run)
fi

echo "Running phase1 finetune"
echo "Stage1 log: $STAGE1_LOG"
"${stage1_cmd[@]}" 2>&1 | tee "$STAGE1_LOG"

if [[ "$STAGE1_DRY_RUN" == "1" ]]; then
    "$PYTHON_BIN" - "$SUMMARY_JSON" "$PHASE1_CONFIG_PATH" "$STAGE1_LOG" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
payload = {
    "mode": "stage1_dry_run",
    "config_path": sys.argv[2],
    "stage1_log": sys.argv[3],
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
    echo "Dry-run complete"
    echo "Summary file: $SUMMARY_JSON"
    exit 0
fi

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

echo "Running strict Stage1 A/B evaluation"
"$PYTHON_BIN" tools/stage1_ab_eval_strict_runner.py \
    --config "$PHASE1_CONFIG_PATH" \
    --dataset "$GOLD_EVAL_DATASET" \
    --adapter-dir "$ADAPTER_DIR" \
    --sample-size "$AB_SAMPLE_SIZE" \
    --seed "$AB_SEED" \
    --max-tokens-cap "$AB_MAX_TOKENS_CAP" \
    --hybrid-visual-threshold "$AB_HYBRID_VISUAL_THRESHOLD" \
    --out-dir "$AB_EVAL_OUT_DIR" \
    --reward-backend "$REWARD_BACKEND"

if [[ ! -f "$AB_REPORT_PATH" ]]; then
    echo "Missing strict A/B report: $AB_REPORT_PATH"
    exit 1
fi

PROMOTE_FLAG=()
if [[ "$PROMOTE_ON_PASS" == "1" ]]; then
    PROMOTE_FLAG=(--promote)
fi

FORCE_POLICY_INIT_FLAG=()
if [[ "$FORCE_POLICY_INIT" == "1" ]]; then
    FORCE_POLICY_INIT_FLAG=(--force-policy-init)
fi
GATE_CONFIG_FLAG=()
if [[ -f "$GATE_CONFIG_PATH" ]]; then
    GATE_CONFIG_FLAG=(--gate-config "$GATE_CONFIG_PATH")
fi

echo "Running strict promotion gate"
"$PYTHON_BIN" -u -m tikz_mlx.cli promote-sft \
    --config "$PHASE1_CONFIG_PATH" \
    --baseline-report "$AB_REPORT_PATH" \
    --candidate-report "$AB_REPORT_PATH" \
    --baseline-key base \
    --candidate-key stage1 \
    --min-compile-delta "$MIN_SUBSTANTIVE_COMPILE_DELTA" \
    --min-schema-delta "$MIN_SUBSTANTIVE_TIKZ_DELTA" \
    --min-candidate-compile-rate "$MIN_STAGE1_SUBSTANTIVE_COMPILE_RATE" \
    --min-candidate-schema-rate "$MIN_STAGE1_SUBSTANTIVE_TIKZ_RATE" \
    --candidate-checkpoint "$STAGE1_OUT" \
    --sft-final-path "$SFT_FINAL_PATH" \
    --policy-init-path "$POLICY_INIT_PATH" \
    --run-id "$STAGE1_RUN_ID" \
    ${PROMOTE_FLAG+"${PROMOTE_FLAG[@]}"} \
    ${FORCE_POLICY_INIT_FLAG+"${FORCE_POLICY_INIT_FLAG[@]}"} \
    ${GATE_CONFIG_FLAG+"${GATE_CONFIG_FLAG[@]}"} \
    | tee "$PROMOTION_RESULT_JSON"

"$PYTHON_BIN" - "$SUMMARY_JSON" "$PHASE1_CONFIG_PATH" "$STAGE1_LOG" "$STAGE1_OUT" "$ADAPTER_DIR" "$AB_REPORT_PATH" "$PROMOTION_RESULT_JSON" "$SFT_FINAL_PATH" "$POLICY_INIT_PATH" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
promotion_path = pathlib.Path(sys.argv[7])

promotion_payload = None
try:
    promotion_payload = json.loads(promotion_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    promotion_payload = None

payload = {
    "mode": "phase1_full",
    "config_path": sys.argv[2],
    "stage1_log": sys.argv[3],
    "stage1_output": sys.argv[4],
    "stage1_adapter_dir": sys.argv[5],
    "strict_ab_report": sys.argv[6],
    "promotion_result": str(promotion_path),
    "sft_final_path": sys.argv[8],
    "policy_init_path": sys.argv[9],
    "gate_passed": bool(promotion_payload and promotion_payload.get("gate", {}).get("passed")),
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "Phase1 finetune complete"
echo "Summary file: $SUMMARY_JSON"
