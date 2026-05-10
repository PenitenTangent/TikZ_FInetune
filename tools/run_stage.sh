#!/bin/bash
# Run a single curriculum stage with checkpoint resumption
# Usage: run_stage.sh <stage_num> [--resume]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  PYTHON_EXE=".venv/bin/python"
else
  PYTHON_EXE="${PYTHON_EXE:-python3}"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Check arguments
if [ $# -lt 1 ]; then
  echo "Usage: run_stage.sh <stage_num> [--resume|--no-resume] [--config <path>] [--resume-from <adapter>]"
  echo ""
  echo "Examples:"
  echo "  ./tools/run_stage.sh 1              # Run stage 1 from scratch"
  echo "  ./tools/run_stage.sh 2 --resume     # Resume stage 2 from checkpoint"
  echo "  ./tools/run_stage.sh 3              # Run stage 3 (resumes if checkpoint exists)"
  exit 1
fi

STAGE_NUM=$1
shift
RESUME_FLAG="--resume"
OVERRIDE_CONFIG=""
RESUME_FROM=""
ALLOW_IMPLICIT_RESUME="${ALLOW_IMPLICIT_RESUME:-0}"
ALLOW_QUARANTINED="${ALLOW_QUARANTINED:-0}"
SKIP_DATA_GATES="${SKIP_DATA_GATES:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --resume)
      RESUME_FLAG="--resume"
      shift
      ;;
    --no-resume)
      RESUME_FLAG="--no-resume"
      shift
      ;;
    --config)
      OVERRIDE_CONFIG="${2:?--config requires a path}"
      shift 2
      ;;
    --resume-from)
      RESUME_FROM="${2:?--resume-from requires an adapter path}"
      shift 2
      ;;
    --allow-implicit-resume)
      ALLOW_IMPLICIT_RESUME="1"
      shift
      ;;
    --allow-quarantined)
      ALLOW_QUARANTINED="1"
      shift
      ;;
    --skip-data-gates)
      SKIP_DATA_GATES="1"
      shift
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Validate stage number
if ! [[ "$STAGE_NUM" =~ ^[1-5]$ ]]; then
  echo "ERROR: Stage number must be 1-5"
  exit 1
fi

# Stage configurations
STAGES=(
  ""  # index 0 unused
  "configs/curriculum_stage1.yaml"
  "configs/curriculum_stage2.yaml"
  "configs/curriculum_stage3.yaml"
  "configs/curriculum_stage4.yaml"
  "configs/curriculum_stage5.yaml"
)

# Checkpoint directories
CHECKPOINT_DIRS=(
  ""  # index 0 unused
  "runs/curriculum_stage1"
  "runs/curriculum_stage2"
  "runs/curriculum_stage3"
  "runs/curriculum_stage4"
  "runs/curriculum_stage5"
)

# Adapter outputs
ADAPTER_OUTPUTS=(
  ""  # index 0 unused
  "runs/tikz_stage1_adapter.safetensors"
  "runs/tikz_stage2_adapter.safetensors"
  "runs/tikz_stage3_adapter.safetensors"
  "runs/tikz_stage4_adapter.safetensors"
  "runs/tikz_stage5_adapter.safetensors"
)

# Detect dry-run mode
DRY_RUN_FLAG=""
if [[ "$TRAIN_EXTRA_ARGS" == *"--dry-run"* ]]; then
  DRY_RUN_FLAG="--dry-run"
fi

config_file="${STAGES[$STAGE_NUM]}"
if [ -n "$OVERRIDE_CONFIG" ]; then
  config_file="$OVERRIDE_CONFIG"
fi
checkpoint_dir="${CHECKPOINT_DIRS[$STAGE_NUM]}"
adapter_output="${CHECKPOINT_DIRS[$STAGE_NUM]}/final_adapter.safetensors"
published_adapter="${ADAPTER_OUTPUTS[$STAGE_NUM]}"
prev_adapter=""
run_id="curriculum_stage${STAGE_NUM}"
stage_iters=$(
  "$PYTHON_EXE" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); print(int(c['training']['iters']))" \
    "$config_file"
)
stage_log="$checkpoint_dir/train_stage${STAGE_NUM}.log"

if [ $STAGE_NUM -gt 1 ]; then
  prev_adapter="${ADAPTER_OUTPUTS[$((STAGE_NUM - 1))]}"
fi

echo "========================================="
echo "Stage $STAGE_NUM Training"
echo "========================================="
echo "Config: $config_file"
echo "Checkpoints: $checkpoint_dir"
echo "Output adapter: $adapter_output"
echo "Published adapter: $published_adapter"
echo "Target iterations: $stage_iters"
echo "Log file: $stage_log"
echo ""

# Create checkpoint directory
mkdir -p "$checkpoint_dir"

# Ensure the resume shim exists at the expected root path. The trainer looks for
# runs/adapter_config.json when resuming from a safetensors adapter, so we write
# a fresh copy from the current stage's LoRA settings before training starts.
"$PYTHON_EXE" - <<'PY' "$config_file" "$PROJECT_ROOT/runs/adapter_config.json"
import json
import pathlib
import sys

import yaml

config_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
with config_path.open("r", encoding="utf-8") as handle:
  config = yaml.safe_load(handle)
training = config.get("training", {})
payload = {
  "rank": int(training.get("lora_rank", 24)),
  "alpha": int(training.get("lora_alpha", 48)),
  "dropout": float(training.get("lora_dropout", 0.0)),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# Check for existing checkpoint
latest_checkpoint=""
if [ -d "$checkpoint_dir" ]; then
  latest_checkpoint=$(ls -t "$checkpoint_dir"/*_adapters.safetensors 2>/dev/null | head -1)
fi

# ── Data quality gates ───────────────────────────────────────────────────────
# Run the mandatory pre-training data pipeline unless explicitly skipped.
# Set SKIP_DATA_GATES=1 or pass --skip-data-gates for resumption workflows where
# data was already gated on a prior invocation.
STAGE_TRAIN_JSONL=""
# Try to find the original raw/source training file for this stage.
# Typically data/prepared/curriculum/train_stageN.jsonl (raw)
for candidate in \
    "data/prepared/curriculum/train_stage${STAGE_NUM}.jsonl" \
    "data/prepared/train_stage${STAGE_NUM}.jsonl" \
    "data/prepared/train.jsonl"; do
  if [ -f "$candidate" ]; then
    STAGE_TRAIN_JSONL="$candidate"
    break
  fi
done

# Read expected clean paths from config
config_paths=$( "$PYTHON_EXE" - <<'PY' "$config_file"
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
t = c.get('training', {})
def get_p(k):
    v = t.get(k)
    return str(v) if v is not None else ""
print(f"CLEAN_JSONL='{get_p('dataset_path')}'")
print(f"PRETOK_OUT='{get_p('pretokenized_cache_path')}'")
print(f"VAL_JSONL='{get_p('val_dataset_path')}'")
print(f"GOLD_JSONL='{get_p('gold_eval_dataset_path')}'")
PY
)
eval "$config_paths"

VAL_FLAG=""
if [ -n "$VAL_JSONL" ]; then
  if [ -f "$VAL_JSONL" ]; then
    VAL_FLAG="--val $VAL_JSONL"
  else
    echo "ERROR: configured val_dataset_path does not exist: $VAL_JSONL"
    exit 1
  fi
fi

GOLD_FLAG=""
if [ -n "$GOLD_JSONL" ]; then
  if [ -f "$GOLD_JSONL" ]; then
    GOLD_FLAG="--gold $GOLD_JSONL"
  else
    echo "ERROR: configured gold_eval_dataset_path does not exist: $GOLD_JSONL"
    exit 1
  fi
fi

# Ensure we don't try to pretokenize if no path is set in config
SKIP_PRETOKENIZE_FLAG=""
if [ -z "$PRETOK_OUT" ]; then
  SKIP_PRETOKENIZE_FLAG="--skip-pretokenize"
fi

if [ "$SKIP_DATA_GATES" = "1" ]; then
  echo "WARNING: Skipping data gates (SKIP_DATA_GATES=1 / --skip-data-gates)."
elif [ -z "$STAGE_TRAIN_JSONL" ]; then
  echo "ERROR: No raw training dataset found for stage $STAGE_NUM."
  echo "       Expected one of: data/prepared/curriculum/train_stage${STAGE_NUM}.jsonl"
  exit 1
else
  echo "Running data quality gates for Stage $STAGE_NUM on: $STAGE_TRAIN_JSONL"
  # shellcheck disable=SC2086
  bash tools/run_data_gates.sh \
    --stage "$STAGE_NUM" \
    --input "$STAGE_TRAIN_JSONL" \
    --clean-output "$CLEAN_JSONL" \
    --pretok-output "$PRETOK_OUT" \
    $SKIP_PRETOKENIZE_FLAG \
    $VAL_FLAG $GOLD_FLAG || {
      echo "ERROR: Data gates failed for Stage $STAGE_NUM. Aborting training."
      exit 1
    }
fi

# Build training command
cmd=(
  "$PYTHON_EXE" -u -m tikz_mlx.cli train
  --config "$config_file"
  --output-path "$adapter_output"
  --run-id "$run_id"
  --iters "$stage_iters"
  --save-interval 100
)

# Determine resume behavior
if [ -n "$RESUME_FROM" ]; then
  if [ ! -f "$RESUME_FROM" ] && [ ! -d "$RESUME_FROM" ]; then
    echo "ERROR: --resume-from path does not exist: $RESUME_FROM"
    exit 1
  fi
  echo "Resuming from explicit adapter: $RESUME_FROM"
  cmd+=(--resume-adapter "$RESUME_FROM")
  
  if [ "$ALLOW_QUARANTINED" != "1" ]; then
      "$PYTHON_EXE" - <<'PY' "$RESUME_FROM" "$PROJECT_ROOT"
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2] + "/src")
from tikz_mlx.quarantine import assert_not_quarantined
adapter = Path(sys.argv[1])
# Handle both file and directory adapter paths
if adapter.is_dir():
    adapter = adapter / "adapters.safetensors"
assert_not_quarantined(adapter)
PY
      if [ $? -ne 0 ]; then
          echo "ERROR: Attempted to resume from a quarantined adapter!"
          exit 1
      fi
  fi
elif [ "$RESUME_FLAG" == "--resume" ]; then
  if [ "$STAGE_NUM" -ge 4 ] && [ "$ALLOW_IMPLICIT_RESUME" != "1" ]; then
    echo "ERROR: Stage >= 4 requires explicit --resume-from or ALLOW_IMPLICIT_RESUME=1 for safety."
    exit 1
  fi
  if [ -n "$latest_checkpoint" ]; then
    echo "Resuming from checkpoint: $latest_checkpoint"
    cmd+=(--resume-adapter "$latest_checkpoint")
  elif [ $STAGE_NUM -gt 1 ] && [ -f "$prev_adapter" ]; then
    echo "No checkpoint found. Resuming from previous stage adapter: $prev_adapter"
    cmd+=(--resume-adapter "$prev_adapter")
  else
    echo "No checkpoint or previous adapter found. Starting fresh."
  fi
elif [ -n "$latest_checkpoint" ]; then
  echo "Warning: Checkpoint exists but --resume not specified."
  echo "  To resume: run_stage.sh $STAGE_NUM --resume"
  echo "  To start fresh: delete $latest_checkpoint"
fi

# Optional passthrough args (example: TRAIN_EXTRA_ARGS="--dry-run --iters 20")
if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra_args=( ${TRAIN_EXTRA_ARGS} )
  cmd+=("${extra_args[@]}")
fi

echo "Running: ${cmd[*]}"
echo ""

# Run training
"$PYTHON_EXE" tools/run_with_live_progress_tqdm.py \
  --label "stage${STAGE_NUM}" \
  --total-iters "$stage_iters" \
  --checkpoint-dir "$checkpoint_dir" \
  --log-file "$stage_log" \
  -- "${cmd[@]}"

train_exit_code=$?

if [ $train_exit_code -ne 0 ]; then
  echo "ERROR: Stage $STAGE_NUM training failed with exit code $train_exit_code"
  exit 1
fi

echo ""
echo "✓ Stage $STAGE_NUM completed successfully"
echo "  Final adapter: $adapter_output"

# Publish a stable adapter path for chaining into next stage.
# 4. Post-training Eval Gate (Mandatory for non-dry-runs)
if [ -z "$DRY_RUN_FLAG" ]; then
    echo ""
    echo "Running post-training evaluation gate..."
    if ./tools/run_eval_gate.sh --config "$config_file" --adapter "$adapter_output"; then
        echo "✓ Evaluation gate passed."
    else
        echo "ERROR: Evaluation gate failed. Adapter is quarantined."
        exit 1
    fi
fi

# 5. Publish adapter if successful
PUBLISH_ON_SUCCESS_DEFAULT="1"
case "$config_file" in
  *recovery*.yaml) PUBLISH_ON_SUCCESS_DEFAULT="0" ;;
esac
if [ "$STAGE_NUM" -ge 4 ]; then
  PUBLISH_ON_SUCCESS_DEFAULT="0"
fi
PUBLISH_ON_SUCCESS="${PUBLISH_ON_SUCCESS:-$PUBLISH_ON_SUCCESS_DEFAULT}"
if [ "$PUBLISH_ON_SUCCESS" != "1" ]; then
  echo "  Publish skipped (PUBLISH_ON_SUCCESS=$PUBLISH_ON_SUCCESS)."
elif [ -f "$adapter_output" ]; then
  mkdir -p "$(dirname "$published_adapter")"
  cp -f "$adapter_output" "$published_adapter"
  echo "  Published adapter: $published_adapter"
else
  echo "  No adapter file produced (likely dry-run); skipping publish copy."
fi

# Clean up old checkpoints to limit disk usage (keep latest 2 + metadata)
if [ -d "$checkpoint_dir" ]; then
  echo "Cleaning old checkpoints in $checkpoint_dir (keeping latest 2)..."
  stale_ckpts=$(ls -t "$checkpoint_dir"/*_adapters.safetensors 2>/dev/null | tail -n +3 || true)
  if [ -n "$stale_ckpts" ]; then
    while IFS= read -r ckpt; do
      [ -z "$ckpt" ] && continue
      rm -f "$ckpt" "${ckpt}.metadata.json"
    done <<< "$stale_ckpts"
  fi
fi

echo ""
echo "Next steps:"
if [ $STAGE_NUM -lt 5 ]; then
  next_stage=$((STAGE_NUM + 1))
  if [ "$next_stage" -ge 4 ]; then
    echo "  - Stage $next_stage is recovery-gated; publish is skipped by default."
    echo "  - Run only after rollback/data/mask gates pass: ./tools/run_stage.sh $next_stage"
  else
    echo "  - Run stage $next_stage: ./tools/run_stage.sh $next_stage"
    echo "  - Or resume stage $next_stage: ./tools/run_stage.sh $next_stage --resume"
  fi
else
  echo "  - All 5 stages completed!"
fi
