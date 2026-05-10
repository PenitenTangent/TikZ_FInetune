#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/clean_adapter_recovery.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_ID="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(cfg["model"]["model_id"])
PY
)"
MAX_CONTEXT_TOKENS="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(cfg["model"]["max_context_tokens"])
PY
)"

echo "=========================================="
echo "1. Prepare Raw Data"
echo "=========================================="
# "$PYTHON_BIN" -m tikz_mlx.cli prepare-dataset --config "$CONFIG_PATH" --dataset-id nllg/DaTikZ-V4 --overwrite

echo "=========================================="
echo "2. Normalize Coordinates on Full Source"
echo "=========================================="
# "$PYTHON_BIN" tools/normalize_coordinates.py --input data/prepared/all_prepared_sft.jsonl --output data/prepared/all_prepared_sft_norm.jsonl

echo "=========================================="
echo "3. Unify Full Source"
echo "=========================================="
# "$PYTHON_BIN" tools/unify_dataset.py --input data/prepared/all_prepared_sft_norm.jsonl --output data/prepared/all_prepared_sft_unified.jsonl

echo "=========================================="
echo "4. Filter Clean Source"
echo "=========================================="
"$PYTHON_BIN" tools/recovery_artifacts.py filter-clean-data \
  --input data/prepared/all_prepared_sft_unified.jsonl \
  --output data/prepared/all_prepared_sft_clean.jsonl \
  --audit-output data/manifests/clean_data_filter_audit.json \
  --max-token-length "$MAX_CONTEXT_TOKENS"

echo "=========================================="
echo "5. Split Clean Dataset"
echo "=========================================="
"$PYTHON_BIN" -m tikz_mlx.cli split-dataset \
  --config "$CONFIG_PATH" \
  --train-path data/prepared/all_prepared_sft_clean.jsonl \
  --stage2-path data/prepared/all_prepared_stage2.jsonl \
  --val-fraction 0.02 \
  --gold-eval-fraction 0.01 \
  --overwrite

echo "=========================================="
echo "6. Build Quality-Balanced Train Set"
echo "=========================================="
"$PYTHON_BIN" tools/recovery_artifacts.py build-mode-balanced-train \
  --input data/prepared/train_unified.jsonl \
  --output data/prepared/train_quality_balanced.jsonl \
  --audit-output data/manifests/mode_balance_audit.json \
  --seed 20260427

echo "=========================================="
echo "7. Validate Recovery Contract and Splits"
echo "=========================================="
"$PYTHON_BIN" tools/recovery_artifacts.py check-contract --dataset data/prepared/train_quality_balanced.jsonl
"$PYTHON_BIN" tools/recovery_artifacts.py check-contract --dataset data/prepared/val_unified.jsonl
"$PYTHON_BIN" tools/recovery_artifacts.py check-contract --dataset data/prepared/gold_eval_unified.jsonl
"$PYTHON_BIN" tools/recovery_artifacts.py check-splits \
  --train data/prepared/train_quality_balanced.jsonl \
  --val data/prepared/val_unified.jsonl \
  --gold data/prepared/gold_eval_unified.jsonl

echo "=========================================="
echo "8. Build Gate Config, Eval Manifest, and Repetition Sidecar"
echo "=========================================="
"$PYTHON_BIN" tools/recovery_artifacts.py write-gate-config --output data/manifests/gate_config_v1.json --overwrite
"$PYTHON_BIN" tools/recovery_artifacts.py build-eval-manifest \
  --dataset data/prepared/gold_eval_unified.jsonl \
  --output data/manifests/eval_manifest_v1.json \
  --seed 20260427 \
  --max-tokens 2048 \
  --decoding-profile clean_adapter_recovery \
  --compiler-config "$CONFIG_PATH"
"$PYTHON_BIN" tools/recovery_artifacts.py build-repetition-sidecar \
  --output data/prepared/repetition_penalty_examples.jsonl \
  --mine-dir outputs \
  --max-examples 200

NORMALIZATION_CONFIG_HASH="$("$PYTHON_BIN" - <<'PY'
import json
payload = {
    "clean_data_filter": json.load(open("data/manifests/clean_data_filter_audit.json", encoding="utf-8")).get("quality_filter_config_hash"),
    "mode_balance": json.load(open("data/manifests/mode_balance_audit.json", encoding="utf-8")).get("mode_balance_config_hash"),
}
import hashlib
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
print(hashlib.sha256(encoded).hexdigest())
PY
)"

echo "=========================================="
echo "9. Pretokenize"
echo "=========================================="
"$PYTHON_BIN" tools/pretokenize_dataset.py \
  --model-id "$MODEL_ID" \
  --dataset data/prepared/train_quality_balanced.jsonl \
  --output data/prepared/train_quality_balanced_tokenized.npy \
  --max-tokens "$MAX_CONTEXT_TOKENS" \
  --prompt-contract-version tikz_partial_decode_v1 \
  --normalization-config-hash "$NORMALIZATION_CONFIG_HASH" \
  --disabled-rules ""

echo "=========================================="
echo "10. Pack Dataset"
echo "=========================================="
"$PYTHON_BIN" tools/pack_tokenized_dataset.py \
  --input data/prepared/train_quality_balanced_tokenized.npy \
  --output data/prepared/train_quality_balanced_packed.npy \
  --max-tokens "$MAX_CONTEXT_TOKENS" \
  --assistant-token 4368 \
  --pretokenize-audit data/prepared/train_quality_balanced_tokenized_audit.json \
  --prompt-contract-version tikz_partial_decode_v1 \
  --normalization-config-hash "$NORMALIZATION_CONFIG_HASH" \
  --disabled-rules "" \
  --scoring-status skipped_plain_ce

echo "=========================================="
echo "Pipeline Complete! Dataset is ready."
echo "=========================================="
