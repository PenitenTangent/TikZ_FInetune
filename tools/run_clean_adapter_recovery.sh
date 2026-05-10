#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/clean_adapter_recovery.yaml}"
TRAIN_DATASET="${TRAIN_DATASET:-data/prepared/train_quality_balanced.jsonl}"
VAL_DATASET="${VAL_DATASET:-data/prepared/val_unified.jsonl}"
GOLD_EVAL_DATASET="${GOLD_EVAL_DATASET:-data/prepared/gold_eval_unified.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-data/manifests/eval_manifest_v1.json}"
GATE_CONFIG_PATH="${GATE_CONFIG_PATH:-data/manifests/gate_config_v1.json}"
if [[ ! -f "$GATE_CONFIG_PATH" ]]; then
    GATE_CONFIG_PATH="configs/gate_config_v1.json"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-clean_adapter_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-runs/logs/${RUN_TAG}}"
RUN_DIR="${RUN_DIR:-runs/${RUN_TAG}}"
RUN_CONFIG="${LOG_DIR}/clean_adapter_recovery.yaml"
OUTPUT_PATH="${RUN_DIR}/clean_adapter.safetensors"
PROMOTE_ON_PASS="${PROMOTE_ON_PASS:-0}"
ITERS="${ITERS:-5000}"


mkdir -p "$LOG_DIR" "$RUN_DIR"

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        echo "Missing ${label}: $path"
        exit 1
    fi
}

require_file "$CONFIG_PATH" "clean adapter config"
require_file "$TRAIN_DATASET" "quality-balanced train dataset"
require_file "$VAL_DATASET" "validation dataset"
require_file "$GOLD_EVAL_DATASET" "gold eval dataset"
require_file "$MANIFEST_PATH" "eval manifest"
require_file "$GATE_CONFIG_PATH" "gate config"

echo "Validating clean data contract and split isolation"
"$PYTHON_BIN" tools/recovery_artifacts.py check-contract --dataset "$TRAIN_DATASET"
"$PYTHON_BIN" tools/recovery_artifacts.py check-contract --dataset "$VAL_DATASET"
"$PYTHON_BIN" tools/recovery_artifacts.py check-contract --dataset "$GOLD_EVAL_DATASET"
"$PYTHON_BIN" tools/recovery_artifacts.py check-splits --train "$TRAIN_DATASET" --val "$VAL_DATASET" --gold "$GOLD_EVAL_DATASET"

echo "Materializing fixed eval sets from manifest"
SENTINEL_DATASET="${LOG_DIR}/sentinel_32.jsonl"
ABLATION_DATASET="${LOG_DIR}/ablation_100.jsonl"
PROMOTION_DATASET="${LOG_DIR}/promotion_120.jsonl"
"$PYTHON_BIN" tools/recovery_artifacts.py materialize-eval-set --dataset "$GOLD_EVAL_DATASET" --manifest "$MANIFEST_PATH" --set-name sentinel_32 --output "$SENTINEL_DATASET"
"$PYTHON_BIN" tools/recovery_artifacts.py materialize-eval-set --dataset "$GOLD_EVAL_DATASET" --manifest "$MANIFEST_PATH" --set-name ablation_100 --output "$ABLATION_DATASET"
"$PYTHON_BIN" tools/recovery_artifacts.py materialize-eval-set --dataset "$GOLD_EVAL_DATASET" --manifest "$MANIFEST_PATH" --set-name promotion_120 --output "$PROMOTION_DATASET"

echo "Writing isolated clean-adapter run config"
"$PYTHON_BIN" - "$CONFIG_PATH" "$RUN_CONFIG" "$RUN_DIR" <<'PY'
import pathlib
import sys
import yaml

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
run_dir = pathlib.Path(sys.argv[3])
data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
root = pathlib.Path.cwd()
data.setdefault("paths", {})["runs_dir"] = str(root / run_dir)
data.setdefault("paths", {})["outputs_dir"] = str(root / f"outputs/{run_dir.name}")
data.setdefault("paths", {})["prepared_dir"] = str(root / "data/prepared")
data.setdefault("paths", {})["manifests_dir"] = str(root / "data/manifests")
data.setdefault("paths", {})["cache_dir"] = str(root / ".cache")

# Absolutize compiler binary path
compiler = data.setdefault("compiler", {})
tectonic = compiler.get("tectonic_binary", "tools/bin/tectonic")
compiler["tectonic_binary"] = str(root / tectonic)

training = data.setdefault("training", {})
training["dataset_path"] = str(root / "data/prepared/train_quality_balanced.jsonl")
training["val_dataset_path"] = str(root / "data/prepared/val_unified.jsonl")
training["gold_eval_dataset_path"] = str(root / "data/prepared/gold_eval_unified.jsonl")
training["pretokenized_packed_cache_path"] = str(root / "data/prepared/train_quality_balanced_packed.npy")
training["learning_rate"] = 2e-5
training["lora_rank"] = 8
training["lora_alpha"] = 16
training["lora_dropout"] = 0.1
training["lora_num_layers"] = 16
training["resume_adapter_path"] = None
training["auto_resume_latest_checkpoint"] = False
training["reward_weighted_loss"] = False
training["reward_weight_path"] = None
training["syntax_weighted_loss"] = False
training["syntax_weight_path"] = None
training["allow_full_training"] = True
training["steps_per_save"] = 100
training["steps_per_eval"] = 100
training["checkpoint_keep_last"] = 40
stage2 = training.setdefault("stage2", {})
stage2["enabled"] = False
stage2["allow_full_training"] = False
# Absolutize stage2 paths too
for key in ("dataset_path", "val_dataset_path", "gold_eval_dataset_path",
            "checkpoint_dir", "reward_cache_dir", "telemetry_path"):
    val = stage2.get(key)
    if isinstance(val, str) and val and not val.startswith("/"):
        stage2[key] = str(root / val)
target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

PY

echo "Running fresh clean-adapter training to ${ITERS} iterations"
"$PYTHON_BIN" -u -m tikz_mlx.cli train \
    --config "$RUN_CONFIG" \
    --dataset "$TRAIN_DATASET" \
    --val-dataset "$VAL_DATASET" \
    --output-path "$OUTPUT_PATH" \
    --run-id "$RUN_TAG" \
    --iters "$ITERS" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo "Running sentinel eval on all staged checkpoints"
SENTINEL_EVAL_DIR="${LOG_DIR}/sentinel_eval"
"$PYTHON_BIN" tools/ab_eval.py \
    --config "$RUN_CONFIG" \
    --dataset "$SENTINEL_DATASET" \
    --checkpoint-dir "$RUN_DIR" \
    --num-samples 32 \
    --seed 1 \
    --max-tokens 2048 \
    --out-dir "$SENTINEL_EVAL_DIR"
"$PYTHON_BIN" tools/recovery_artifacts.py check-ab-gate \
    --results "${SENTINEL_EVAL_DIR}/results.json" \
    --candidate-key iter_200 \
    --gate sentinel \
    --gate-config "$GATE_CONFIG_PATH"

echo "Running ablation eval on all staged checkpoints"
ABLATION_EVAL_DIR="${LOG_DIR}/ablation_eval"
"$PYTHON_BIN" tools/ab_eval.py \
    --config "$RUN_CONFIG" \
    --dataset "$ABLATION_DATASET" \
    --checkpoint-dir "$RUN_DIR" \
    --num-samples 100 \
    --seed 1 \
    --max-tokens 2048 \
    --out-dir "$ABLATION_EVAL_DIR"

CANDIDATE_JSON="${LOG_DIR}/staged_candidate.json"
"$PYTHON_BIN" - "$ABLATION_EVAL_DIR/results.json" "$GATE_CONFIG_PATH" "$RUN_DIR" "$CANDIDATE_JSON" "$ITERS" <<'PY'

import json
import pathlib
import sys

sys.path.insert(0, "src")
from tikz_mlx.recovery import evaluate_ab_result_gate

gate_path = pathlib.Path(sys.argv[2])
run_dir = pathlib.Path(sys.argv[3])
out_path = pathlib.Path(sys.argv[4])
max_iters = int(sys.argv[5])
results_path = pathlib.Path(sys.argv[1])
payload = json.loads(results_path.read_text(encoding="utf-8"))
gate_config = json.loads(gate_path.read_text(encoding="utf-8"))
base_milestones = [500, 1000, 2000, 3000, 5000]
iterations = [i for i in base_milestones if i <= max_iters]
if max_iters not in iterations:
    iterations.append(max_iters)
if f"iter_{max_iters}" not in payload:
    raise SystemExit(f"The {max_iters} checkpoint must be evaluated before candidate selection.")

eligible = []
checks = {}
for iteration in iterations:
    key = f"iter_{iteration}"
    result = evaluate_ab_result_gate(payload, candidate_key=key, gate="ablation", gate_config=gate_config)
    checks[key] = result
    if result["passed"]:
        block = payload[key]
        eligible.append(
            (
                float(block.get("repetition_loop_rate", 1.0)),
                float(block.get("truncation_rate", 1.0)),
                -float(block.get("compile_rate", 0.0)),
                -iteration,
                iteration,
            )
        )
if not eligible:
    out_path.write_text(json.dumps({"eligible": [], "checks": checks}, indent=2), encoding="utf-8")
    raise SystemExit("No staged checkpoint passed ablation gate.")
eligible.sort()
iteration = eligible[0][-1]
checkpoint = run_dir / f"{iteration:07d}_adapters.safetensors"
if not checkpoint.exists():
    raise SystemExit(f"Selected checkpoint missing: {checkpoint}")
payload_out = {"iteration": iteration, "checkpoint": str(checkpoint), "checks": checks}
out_path.write_text(json.dumps(payload_out, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload_out, indent=2, sort_keys=True))
PY

CANDIDATE_CHECKPOINT="$("$PYTHON_BIN" - "$CANDIDATE_JSON" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])
PY
)"

echo "Running promotion_120 strict eval from staged candidate"
ADAPTER_DIR="${LOG_DIR}/candidate_adapter_dir"
mkdir -p "$ADAPTER_DIR"
cp "$RUN_DIR/adapter_config.json" "$ADAPTER_DIR/adapter_config.json"
cp "$CANDIDATE_CHECKPOINT" "$ADAPTER_DIR/adapters.safetensors"
PROMOTION_EVAL_DIR="${LOG_DIR}/promotion_eval"
"$PYTHON_BIN" tools/stage1_ab_eval_strict_runner.py \
    --config "$RUN_CONFIG" \
    --dataset "$PROMOTION_DATASET" \
    --adapter-dir "$ADAPTER_DIR" \
    --sample-size 120 \
    --seed 1 \
    --max-tokens-cap 2048 \
    --hybrid-visual-threshold 0.75 \
    --out-dir "$PROMOTION_EVAL_DIR" \
    --reward-backend emd

PROMOTE_FLAG=()
if [[ "$PROMOTE_ON_PASS" == "1" ]]; then
    PROMOTE_FLAG=(--promote)
fi

echo "Running promotion gate"
"$PYTHON_BIN" -u -m tikz_mlx.cli promote-sft \
    --config "$RUN_CONFIG" \
    --baseline-report "$PROMOTION_EVAL_DIR/report.json" \
    --candidate-report "$PROMOTION_EVAL_DIR/report.json" \
    --baseline-key base \
    --candidate-key stage1 \
    --min-compile-delta 0.0 \
    --min-schema-delta 0.0 \
    --min-candidate-compile-rate 0.20 \
    --min-candidate-schema-rate 0.75 \
    --candidate-checkpoint "$CANDIDATE_CHECKPOINT" \
    --sft-final-path runs/sft_final.safetensors \
    --policy-init-path runs/policy_init.safetensors \
    --run-id "$RUN_TAG" \
    --gate-config "$GATE_CONFIG_PATH" \
    ${PROMOTE_FLAG+"${PROMOTE_FLAG[@]}"} \
    | tee "${LOG_DIR}/promotion_result.json"

echo "Clean adapter recovery run complete"
echo "Candidate: $CANDIDATE_CHECKPOINT"
echo "Logs: $LOG_DIR"
