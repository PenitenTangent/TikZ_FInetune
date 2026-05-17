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
export PYTHON_EXE
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PROJECT_ROOT/tools/resume_offset.sh"

# Check arguments
if [ $# -lt 1 ]; then
  echo "Usage: run_stage.sh <stage_num> [--resume|--no-resume] [--config <path>] [--resume-from <adapter>] [--skip-gate]"
  echo ""
  echo "Examples:"
  echo "  ./tools/run_stage.sh 0              # Run synthetic primitive warmup from scratch"
  echo "  ./tools/run_stage.sh 1              # Run stage 1, resuming from stage 0 if published"
  echo "  ./tools/run_stage.sh 3              # Run stage 3 (resumes if checkpoint exists)"
  echo "  ./tools/run_stage.sh 0 --skip-gate  # Run stage 0 without promotion gate check"
  exit 1
fi

STAGE_NUM=$1
shift
RESUME_FLAG="" # Will be defaulted based on stage number
OVERRIDE_CONFIG=""
RESUME_FROM=""
ALLOW_IMPLICIT_RESUME="${ALLOW_IMPLICIT_RESUME:-0}"
ALLOW_QUARANTINED="${ALLOW_QUARANTINED:-0}"
SKIP_DATA_GATES="${SKIP_DATA_GATES:-0}"
SKIP_GATE="${SKIP_GATE:-0}"

# Set default resume flag: Stage 0 starts from scratch, others resume.
if [ "$STAGE_NUM" = "0" ]; then
  RESUME_FLAG="--no-resume"
else
  RESUME_FLAG="--resume"
fi


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
    --skip-gate)
      SKIP_GATE="1"
      shift
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Validate stage number
if ! [[ "$STAGE_NUM" =~ ^[0-5]$ ]]; then
  echo "ERROR: Stage number must be 0-5"
  exit 1
fi

# Safety guard: Stage 0 must not implicitly resume.
if [ "$STAGE_NUM" = "0" ] && [ "$RESUME_FLAG" = "--resume" ] && [ -z "$RESUME_FROM" ]; then
  echo "ERROR: Stage 0 must not implicitly resume. Use --resume-from explicitly if intentional."
  exit 1
fi

# Stage configurations
STAGES=(
  "configs/curriculum_stage0.yaml"
  "configs/curriculum_stage1.yaml"
  "configs/curriculum_stage2.yaml"
  "configs/curriculum_stage3.yaml"
  "configs/curriculum_stage4.yaml"
  "configs/curriculum_stage5.yaml"
)

# Checkpoint directories
CHECKPOINT_DIRS=(
  "runs/curriculum_stage0"
  "runs/curriculum_stage1"
  "runs/curriculum_stage2"
  "runs/curriculum_stage3"
  "runs/curriculum_stage4"
  "runs/curriculum_stage5"
)

# Adapter outputs
ADAPTER_OUTPUTS=(
  "runs/published/stage0"
  "runs/published/stage1"
  "runs/published/stage2"
  "runs/published/stage3"
  "runs/published/stage4"
  "runs/published/stage5"
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
stage_log="$checkpoint_dir/train_stage${STAGE_NUM}.log"

if [ $STAGE_NUM -gt 0 ]; then
  prev_adapter="${ADAPTER_OUTPUTS[$((STAGE_NUM - 1))]}"
fi

# Initial banner (stage_iters calculated later)
echo "========================================="
echo "Stage $STAGE_NUM Training"
echo "========================================="
echo "Config: $config_file"
echo "Checkpoints: $checkpoint_dir"
echo "Output adapter: $adapter_output"
echo "Published adapter: $published_adapter"
echo "Log file: $stage_log"
echo ""

# Create checkpoint directory
mkdir -p "$checkpoint_dir"

# Ensure adapter_config.json exists next to stage checkpoints. Do not write a
# shared runs/adapter_config.json; published adapters use per-stage directories.
"$PYTHON_EXE" - <<'PY' "$config_file" "$checkpoint_dir/adapter_config.json"
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
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# Check for existing checkpoint
latest_checkpoint=""
if [ -d "$checkpoint_dir" ]; then
  # Prioritize the 'last_probe_pass' checkpoint if it exists (safe recovery)
  if [ -f "$checkpoint_dir/last_probe_pass_adapters.safetensors" ]; then
    latest_checkpoint="$checkpoint_dir/last_probe_pass_adapters.safetensors"
    echo "INFO: Found safe checkpoint from last successful probe pass."
  else
    latest_checkpoint=$(ls -t "$checkpoint_dir"/[0-9]*_adapters.safetensors 2>/dev/null | head -1)
  fi
fi

# ── Data quality gates ───────────────────────────────────────────────────────
# Run the mandatory pre-training data pipeline unless explicitly skipped.
# Set SKIP_DATA_GATES=1 or pass --skip-data-gates for resumption workflows where
# data was already gated on a prior invocation.
STAGE_TRAIN_JSONL=""
# Try to find the original raw/source training file for this stage.
# Typically data/prepared/curriculum/train_stageN.jsonl (raw)
for candidate in \
    "data/prepared/curriculum/stage${STAGE_NUM}_compile_curriculum.jsonl" \
    "data/prepared/curriculum/train_stage${STAGE_NUM}.jsonl" \
    "data/prepared/curriculum/synthetic_primitives.jsonl" \
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

REPAIR_CONTRACT_FLAG=""
if [ "$STAGE_NUM" -gt 1 ]; then
  REPAIR_CONTRACT_FLAG="--repair-contract"
fi

if [ "$SKIP_DATA_GATES" = "1" ]; then
  if [ "${ALLOW_SKIP_DATA_GATES:-0}" != "1" ]; then
    echo "ERROR: --skip-data-gates requires ALLOW_SKIP_DATA_GATES=1 environment variable."
    exit 1
  fi
  echo "WARNING: Skipping data gates as requested (ALLOW_SKIP_DATA_GATES=1)."
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
    $REPAIR_CONTRACT_FLAG \
    $SKIP_PRETOKENIZE_FLAG \
    $VAL_FLAG $GOLD_FLAG || {
      echo "ERROR: Data gates failed for Stage $STAGE_NUM. Aborting training."
      exit 1
    }
fi

# ── Iteration Count Calculation ─────────────────────────────────────────────
# We dynamically calculate the number of iterations based on the actual number
# of records in the cleaned dataset after the gates. This ensures a full pass.
coverage_enabled=$(
  "$PYTHON_EXE" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); print('1' if c.get('training', {}).get('coverage', {}).get('enabled', False) else '0')" \
    "$config_file"
)
if [ "$coverage_enabled" = "1" ]; then
  echo "Strict coverage: enabled (resume position comes from coverage_state, not checkpoint filename)"
else
  echo "Strict coverage: disabled"
fi

if [ -f "$CLEAN_JSONL" ]; then
    CLEAN_ROWS=$(wc -l < "$CLEAN_JSONL" | tr -d ' ')
    GRAD_ACCUM=$( "$PYTHON_EXE" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); print(int(c.get('memory', {}).get('gradient_accumulation_steps', 4)))" "$config_file" )

    if [ "$coverage_enabled" = "1" ]; then
        if [ $((CLEAN_ROWS % GRAD_ACCUM)) -ne 0 ]; then
            echo "ERROR: strict no-repeat coverage refuses to round $CLEAN_ROWS rows up to grad_accum=$GRAD_ACCUM."
            echo "       Regenerate stage partitions or choose a gradient_accumulation_steps value that divides the clean row count."
            exit 1
        fi
        stage_iters="$CLEAN_ROWS"
        echo "Strict no-repeat iterations based on $CLEAN_JSONL: $stage_iters (rows=$CLEAN_ROWS, grad_accum=$GRAD_ACCUM)"
    else
        # Round up to multiple of GRAD_ACCUM to ensure the last gradients are applied.
        stage_iters=$(( ((CLEAN_ROWS + GRAD_ACCUM - 1) / GRAD_ACCUM) * GRAD_ACCUM ))
        echo "Dynamic iterations based on $CLEAN_JSONL: $stage_iters (rows=$CLEAN_ROWS, grad_accum=$GRAD_ACCUM)"
    fi
else
    # Fallback to YAML if clean file doesn't exist (should not happen if gates pass)
    stage_iters=$(
      "$PYTHON_EXE" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); print(int(c['training']['iters']))" \
        "$config_file"
    )
    echo "Using iterations from config: $stage_iters"
fi

save_interval=$(
  "$PYTHON_EXE" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); print(int(c.get('training', {}).get('steps_per_save', 1000)))" \
    "$config_file"
)
echo "Save interval: $save_interval"

checkpoint_keep_last=$(
  "$PYTHON_EXE" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); print(max(0, int(c.get('training', {}).get('checkpoint_keep_last', 5))))" \
    "$config_file"
)
echo "Checkpoint retention: keep latest $checkpoint_keep_last"

# Build training command
cmd=(
  "$PYTHON_EXE" -u -m tikz_mlx.cli train
  --config "$config_file"
  --output-path "$adapter_output"
  --run-id "$run_id"
  --iters "$stage_iters"
  --save-interval "$save_interval"
)

append_resume_offset_if_checkpoint() {
  local checkpoint_path="$1"
  if [ "$coverage_enabled" = "1" ]; then
    echo "Strict coverage enabled; skipping filename-derived resume offset for $checkpoint_path"
    return
  fi
  local resume_offset
  resume_offset=$(resume_offset_from_checkpoint_path "$checkpoint_path")
  echo "Detected resumption offset: $resume_offset"
  if [ "$resume_offset" -gt 0 ]; then
    cmd+=(--resume-offset "$resume_offset")
  fi
}

# Determine resume behavior
if [ -n "$RESUME_FROM" ]; then
  if [ ! -f "$RESUME_FROM" ] && [ ! -d "$RESUME_FROM" ]; then
    echo "ERROR: --resume-from path does not exist: $RESUME_FROM"
    exit 1
  fi
  echo "Resuming from explicit adapter: $RESUME_FROM"
  cmd+=(--resume-adapter "$RESUME_FROM")
  append_resume_offset_if_checkpoint "$RESUME_FROM"
  
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
    append_resume_offset_if_checkpoint "$latest_checkpoint"
  elif [ $STAGE_NUM -gt 0 ] && { [ -f "$prev_adapter" ] || [ -d "$prev_adapter" ]; }; then
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

selected_adapter="$adapter_output"

# Publish a stable adapter path for chaining into next stage.
# 4. Post-training stage gate and checkpoint selection (mandatory for non-dry-runs)
if [ -z "$DRY_RUN_FLAG" ] && [ "${SKIP_GATE:-0}" != "1" ]; then
    echo ""
    echo "Running post-training stage gate and selecting last passing checkpoint..."
    selection_json="$checkpoint_dir/selected_checkpoint.json"
    stage_gate_num_samples_default=100
    stage_gate_max_candidates_default=4
    if [ "$STAGE_NUM" = "0" ]; then
        stage_gate_num_samples_default=16
        stage_gate_max_candidates_default=1
    fi
    if "$PYTHON_EXE" tools/select_last_good_checkpoint.py \
        --config "$config_file" \
        --stage "stage${STAGE_NUM}" \
        --gate-config "configs/promotion_gate.yaml" \
        --preferred-adapter "$adapter_output" \
        --checkpoint-dir "$checkpoint_dir" \
        --out "$selection_json" \
        --num-samples "${STAGE_GATE_NUM_SAMPLES:-$stage_gate_num_samples_default}" \
        --max-candidates "${STAGE_GATE_MAX_CANDIDATES:-$stage_gate_max_candidates_default}"; then
        selected_adapter=$("$PYTHON_EXE" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['selected_checkpoint_path'])" "$selection_json")
        echo "✓ Stage gate passed."
        echo "  Selected adapter: $selected_adapter"
    else
        if [ "$STAGE_NUM" = "0" ]; then
            echo "WARNING: Stage 0 gate failed, but continuing as requested."
            selected_adapter="$adapter_output"
        else
            echo "ERROR: No checkpoint passed the stage gate. Adapter(s) have been quarantined as applicable."
            exit 1
        fi
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
elif [ -f "$selected_adapter" ]; then
  mkdir -p "$published_adapter"
  cp -f "$selected_adapter" "$published_adapter/adapters.safetensors"
  cp -f "$checkpoint_dir/adapter_config.json" "$published_adapter/adapter_config.json"
  if [ -f "${selected_adapter}.metadata.json" ]; then
    cp -f "${selected_adapter}.metadata.json" "$published_adapter/checkpoint_metadata.json"
  fi
  echo "  Published adapter: $published_adapter"
else
  echo "  No adapter file produced (likely dry-run); skipping publish copy."
fi

# Clean up old checkpoints to limit disk usage.
if [ -d "$checkpoint_dir" ]; then
  echo "Cleaning old checkpoints in $checkpoint_dir (keeping latest $checkpoint_keep_last)..."
  stale_ckpts=$(ls -t "$checkpoint_dir"/*_adapters.safetensors 2>/dev/null | tail -n +"$((checkpoint_keep_last + 1))" || true)
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
