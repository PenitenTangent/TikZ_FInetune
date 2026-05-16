#!/bin/bash
# Run 5-stage curriculum training without packing
# Each stage resumes from its previous checkpoint
# Old checkpoints are cleaned up to save disk space

set -e

PROJECT_ROOT="/Users/andrisoueslati/Code/TikZ"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  PYTHON_EXE=".venv/bin/python"
else
  PYTHON_EXE="${PYTHON_EXE:-python3}"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Stage configurations
STAGES=(
  "configs/curriculum_stage1.yaml"
  "configs/curriculum_stage2.yaml"
  "configs/curriculum_stage3.yaml"
  "configs/curriculum_stage4.yaml"
  "configs/curriculum_stage5.yaml"
)

# Checkpoint directories (one per stage)
CHECKPOINT_DIRS=(
  "runs/curriculum_stage1"
  "runs/curriculum_stage2"
  "runs/curriculum_stage3"
  "runs/curriculum_stage4"
  "runs/curriculum_stage5"
)

# Final adapter outputs
ADAPTER_OUTPUTS=(
  "runs/tikz_stage1_adapter.safetensors"
  "runs/tikz_stage2_adapter.safetensors"
  "runs/tikz_stage3_adapter.safetensors"
  "runs/tikz_stage4_adapter.safetensors"
  "runs/tikz_stage5_adapter.safetensors"
)

echo "========================================="
echo "5-Stage Curriculum Training (No Packing)"
echo "========================================="

MAX_STAGE="${MAX_STAGE:-3}"
if ! [[ "$MAX_STAGE" =~ ^[1-5]$ ]]; then
  echo "ERROR: MAX_STAGE must be between 1 and 5"
  exit 1
fi
if [ "$MAX_STAGE" -lt 5 ]; then
  echo "Default safety stop: running through stage $MAX_STAGE only."
  echo "Set MAX_STAGE=5 only after Stage4 recovery gates pass."
fi

for stage_idx in "${!STAGES[@]}"; do
  stage_num=$((stage_idx + 1))
  if [ "$stage_num" -gt "$MAX_STAGE" ]; then
    echo "Skipping stage $stage_num because MAX_STAGE=$MAX_STAGE."
    continue
  fi
  config_file="${STAGES[$stage_idx]}"
  checkpoint_dir="${CHECKPOINT_DIRS[$stage_idx]}"
  adapter_output="${CHECKPOINT_DIRS[$stage_idx]}/final_adapter.safetensors"
  published_adapter="${ADAPTER_OUTPUTS[$stage_idx]}"
  run_id="curriculum_stage${stage_num}"
  stage_iters=$(
    "$PYTHON_EXE" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); print(int(c['training']['iters']))" \
      "$config_file"
  )
  stage_log="$checkpoint_dir/train_stage${stage_num}.log"
  prev_adapter=""
  
  if [ $stage_num -gt 1 ]; then
    prev_adapter="${ADAPTER_OUTPUTS[$((stage_idx - 1))]}"
  fi

  echo ""
  echo "============================================"
  echo "Stage $stage_num: $config_file"
  echo "============================================"
  echo "Checkpoints dir: $checkpoint_dir"
  echo "Output adapter: $adapter_output"
  echo "Published adapter: $published_adapter"
  echo "Target iterations: $stage_iters"
  echo "Log file: $stage_log"
  
  # Create checkpoint directory if it doesn't exist
  mkdir -p "$checkpoint_dir"

  # Keep the root adapter shim aligned with the stage LoRA settings. The
  # training planner uses this when resuming from a bare .safetensors file.
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

  # Discover latest stage checkpoint if any
  latest_checkpoint=$(ls -t "$checkpoint_dir"/*_adapters.safetensors 2>/dev/null | head -1 || true)
  
  # Build training command
  cmd=(
    "$PYTHON_EXE" -u -m tikz_mlx.cli train
    --config "$config_file"
    --output-path "$adapter_output"
    --run-id "$run_id"
    --iters "$stage_iters"
    --skip-post-ab-eval
  )
  
  # Resume stage from its own latest checkpoint if available.
  if [ -n "$latest_checkpoint" ]; then
    echo "Resuming from latest checkpoint: $latest_checkpoint"
    cmd+=(--resume-adapter "$latest_checkpoint")
  # Otherwise resume from previous stage adapter if available.
  elif [ $stage_num -gt 1 ] && [ -f "$prev_adapter" ]; then
    echo "Resuming from previous stage adapter: $prev_adapter"
    cmd+=(--resume-adapter "$prev_adapter")
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
    --label "stage${stage_num}" \
    --total-iters "$stage_iters" \
    --checkpoint-dir "$checkpoint_dir" \
    --log-file "$stage_log" \
    -- "${cmd[@]}"
  
  train_exit_code=$?
  
  if [ $train_exit_code -ne 0 ]; then
    echo "ERROR: Stage $stage_num training failed with exit code $train_exit_code"
    exit 1
  fi
  
  echo ""
  echo "✓ Stage $stage_num completed successfully"
  echo "  Final adapter: $adapter_output"

  # Publish stable adapter path for chaining and downstream scripts.
  PUBLISH_ON_SUCCESS_DEFAULT="1"
  if [ "$stage_num" -ge 4 ]; then
    PUBLISH_ON_SUCCESS_DEFAULT="0"
  fi
  PUBLISH_ON_SUCCESS="${PUBLISH_ON_SUCCESS:-$PUBLISH_ON_SUCCESS_DEFAULT}"
  if [ "$PUBLISH_ON_SUCCESS" != "1" ]; then
    echo "  Publish skipped (PUBLISH_ON_SUCCESS=$PUBLISH_ON_SUCCESS)."
  elif [ -f "$adapter_output" ]; then
    mkdir -p "$(dirname "$published_adapter")"
    cp -f "$adapter_output" "$published_adapter"
    if [ -f "${adapter_output}.metadata.json" ]; then
      cp -f "${adapter_output}.metadata.json" "${published_adapter}.metadata.json"
    fi
    echo "  Published adapter: $published_adapter"
  else
    echo "  No adapter file produced (likely dry-run); skipping publish copy."
  fi
  
  # Clean up old checkpoints to save disk space (keep latest only)
  if [ -d "$checkpoint_dir" ]; then
    echo "Cleaning up old checkpoints in $checkpoint_dir (keeping latest 2)..."
    stale_ckpts=$(ls -t "$checkpoint_dir"/*_adapters.safetensors 2>/dev/null | tail -n +3 || true)
    if [ -n "$stale_ckpts" ]; then
      while IFS= read -r ckpt; do
        [ -z "$ckpt" ] && continue
        rm -f "$ckpt" "${ckpt}.metadata.json"
      done <<< "$stale_ckpts"
    fi
    echo "✓ Old checkpoints cleaned"
  fi
  
  echo ""
done

echo ""
echo "========================================="
echo "✓ Curriculum stages 1-$MAX_STAGE completed successfully!"
echo "========================================="
echo ""
echo "Final adapters:"
for i in "${!ADAPTER_OUTPUTS[@]}"; do
  stage_num=$((i + 1))
  adapter="${ADAPTER_OUTPUTS[$i]}"
  if [ -f "$adapter" ]; then
    size=$(du -h "$adapter" | cut -f1)
    echo "  Stage $stage_num: $adapter ($size)"
  fi
done
