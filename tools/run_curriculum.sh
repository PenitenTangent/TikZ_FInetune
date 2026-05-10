#!/bin/bash
# Requires bash. macOS /bin/bash is often 3.2 — avoid ${PIPESTATUS[i]:-word} on PIPESTATUS
# elements (use plain subscripts + defaults); Bash 4+ syntax can trigger parse errors there.
#
# Overrides (optional):
#   TRAIN_DATASET   — train JSONL path (default: data/prepared/train_quality_balanced.jsonl)
#   VAL_DATASET     — validation JSONL path (default: data/prepared/val_unified.jsonl)
#   CURRICULUM_MAX_EXAMPLES — pass --max-examples to build_curriculum (cap rows, file order)
#   PYTHON_BIN      — Python executable (default: .venv/bin/python)
#
# Quick smoke without editing files:
#   CURRICULUM_MAX_EXAMPLES=50 bash tools/run_curriculum.sh 120
# Or slice train data to a temp file, then:
#   head -n 50 data/prepared/train_quality_balanced.jsonl > /tmp/train_curriculum_smoke.jsonl
#   TRAIN_DATASET=/tmp/train_curriculum_smoke.jsonl bash tools/run_curriculum.sh 120
set -euo pipefail

# Configuration
PYTHON_BIN=${PYTHON_BIN:-".venv/bin/python"}
MASTER_CONFIG="configs/clean_adapter_recovery.yaml"
TRAIN_DATASET="${TRAIN_DATASET:-data/prepared/train_quality_balanced.jsonl}"
VAL_DATASET="${VAL_DATASET:-data/prepared/val_unified.jsonl}"
CURRICULUM_DIR="data/prepared/curriculum"
TOTAL_ITERS="${1:-${TOTAL_ITERS:-1500}}"
OUTPUT_ROOT="runs/curriculum_run"
RUN_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_ROOT}/curriculum_${RUN_TAG}"

echo "=========================================================="
echo " Starting Smart Curriculum Pipeline"
echo " Total Iterations Budget: ${TOTAL_ITERS}"
echo " Train data: ${TRAIN_DATASET}"
if [ -n "${CURRICULUM_MAX_EXAMPLES:-}" ]; then
  echo " Curriculum cap: ${CURRICULUM_MAX_EXAMPLES} rows (--max-examples)"
fi
echo "=========================================================="

# Optional RL: master config training.stage2.enabled, or override with RUN_RL=0|1
STAGE2_ENABLED=$(
    "$PYTHON_BIN" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); e=c.get('training',{}).get('stage2',{}).get('enabled',False); print('true' if e else 'false')" \
        "$MASTER_CONFIG"
)
if [ -n "${RUN_RL:-}" ]; then
  case "$(echo "$RUN_RL" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) STAGE2_ENABLED=true ;;
    0|false|no|off) STAGE2_ENABLED=false ;;
  esac
fi

# Step 1: Build Curriculum
echo ""
echo "[1/4] Building Curriculum (K-Means Clustering)..."
CURRICULUM_EXTRA_ARGS=()
if [ -n "${CURRICULUM_MAX_EXAMPLES:-}" ]; then
  CURRICULUM_EXTRA_ARGS+=(--max-examples "$CURRICULUM_MAX_EXAMPLES")
fi

"$PYTHON_BIN" tools/build_curriculum.py \
    --dataset "$TRAIN_DATASET" \
    --val-dataset "$VAL_DATASET" \
    --config "$MASTER_CONFIG" \
    --out-dir "$CURRICULUM_DIR" \
    --iters "$TOTAL_ITERS" \
    --python "$PYTHON_BIN" \
    "${CURRICULUM_EXTRA_ARGS[@]}"

mkdir -p "$RUN_DIR"
cp "$MASTER_CONFIG" "${RUN_DIR}/master_config_backup.yaml"

# Helper to run an SFT curriculum stage (phases 1–3)
run_sft_stage() {
    local STAGE_NUM=$1
    local RESUME_ADAPTER=$2
    local CONFIG="configs/curriculum_stage${STAGE_NUM}.yaml"
    local LOG_FILE="${RUN_DIR}/stage${STAGE_NUM}_train.log"
    
    echo ""
    echo "=========================================================="
    echo " [SFT ${STAGE_NUM}/3] Training curriculum stage ${STAGE_NUM}"
    echo " Config: $CONFIG"
    if [ -n "$RESUME_ADAPTER" ]; then
        echo " Resuming from: $RESUME_ADAPTER"
    fi
    echo "=========================================================="
    
    local STAGE_ITERS
    STAGE_ITERS=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['iters'])")
    
    local CMD=(
        "$PYTHON_BIN" -u -m tikz_mlx.cli train
        --config "$CONFIG"
        --output-path "${RUN_DIR}/stage${STAGE_NUM}/clean_adapter.safetensors"
        --run-id "stage${STAGE_NUM}"
        --iters "$STAGE_ITERS"
        --skip-post-ab-eval
    )
    
    if [ -n "$RESUME_ADAPTER" ]; then
        CMD+=(--resume-adapter "$RESUME_ADAPTER")
    fi
    
    mkdir -p "${RUN_DIR}/stage${STAGE_NUM}"
    set +o pipefail
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    local CMD_STATUS TEE_STATUS
    CMD_STATUS=1
    TEE_STATUS=0
    if [ "${#PIPESTATUS[@]}" -ge 1 ]; then
        CMD_STATUS=${PIPESTATUS[0]}
    fi
    if [ "${#PIPESTATUS[@]}" -ge 2 ]; then
        TEE_STATUS=${PIPESTATUS[1]}
    fi
    set -o pipefail
    if [ "$CMD_STATUS" -ne 0 ]; then
        return "$CMD_STATUS"
    fi
    if [ "$TEE_STATUS" -ne 0 ]; then
        echo "Warning: logging stream failed for stage ${STAGE_NUM} (tee exit ${TEE_STATUS})." >&2
    fi
}

# Step 2: Phase 1 (Foundation)
run_sft_stage 1 ""
PHASE1_ADAPTER="${RUN_DIR}/stage1/clean_adapter.safetensors"

# Step 3: Phase 2 (Intermediate)
if [ ! -f "$PHASE1_ADAPTER" ]; then
    echo "Error: Phase 1 failed to produce adapter weights at $PHASE1_ADAPTER"
    exit 1
fi
run_sft_stage 2 "$PHASE1_ADAPTER"
PHASE2_ADAPTER="${RUN_DIR}/stage2/clean_adapter.safetensors"

# Step 4: Phase 3 (Mastery)
if [ ! -f "$PHASE2_ADAPTER" ]; then
    echo "Error: Phase 2 failed to produce adapter weights at $PHASE2_ADAPTER"
    exit 1
fi
run_sft_stage 3 "$PHASE2_ADAPTER"
PHASE3_ADAPTER="${RUN_DIR}/stage3/clean_adapter.safetensors"

if [ ! -f "$PHASE3_ADAPTER" ]; then
    echo "Error: Phase 3 failed to produce adapter weights at $PHASE3_ADAPTER"
    exit 1
fi

# Step 4.5: One-time final A/B eval on all source examples (base + final adapter)
echo ""
echo "[4/4] One-time final A/B evaluation on full source dataset..."
AB_EVAL_SCRIPT="tools/ab_eval.py"
if [ -f "$AB_EVAL_SCRIPT" ]; then
    AB_SAMPLES=$("$PYTHON_BIN" -c "from pathlib import Path; p=Path('$TRAIN_DATASET'); print(sum(1 for _ in p.open('r', encoding='utf-8')) if p.exists() else 0)")
    if [ "${AB_SAMPLES}" -gt 0 ]; then
        "$PYTHON_BIN" "$AB_EVAL_SCRIPT" \
            --config "$MASTER_CONFIG" \
            --dataset "$TRAIN_DATASET" \
            --adapter-path "$PHASE3_ADAPTER" \
            --checkpoint-dir "${RUN_DIR}/stage3" \
            --num-samples "$AB_SAMPLES" \
            --seed 42 \
            --max-tokens 2048
    else
        echo "Warning: could not compute dataset size for $TRAIN_DATASET; skipping final A/B eval." >&2
    fi
else
    echo "Warning: $AB_EVAL_SCRIPT not found; skipping final A/B eval." >&2
fi

# Step 5: Stage-2 RL (DRGRPO), optional
FINAL_RL_ADAPTER=""
if [ "$STAGE2_ENABLED" = "true" ]; then
    echo ""
    echo "[4/4] Stage-2 RL (DRGRPO)..."
    mkdir -p "${RUN_DIR}/stage4"
    STAGE2_ITERS=$("$PYTHON_BIN" -c "import yaml; c=yaml.safe_load(open('$MASTER_CONFIG')); print(c['training']['stage2']['iters'])")
    RL_LOG="${RUN_DIR}/stage4_rl_train.log"
    RL_CMD=(
        "$PYTHON_BIN" -u -m tikz_mlx.cli train-stage2
        --config "$MASTER_CONFIG"
        --output-path "${RUN_DIR}/stage4/stage2_adapter.safetensors"
        --resume-adapter "$PHASE3_ADAPTER"
        --run-id "stage4_rl"
        --iters "$STAGE2_ITERS"
    )
    set +o pipefail
    "${RL_CMD[@]}" 2>&1 | tee "$RL_LOG"
    RL_CMD_STATUS=1
    RL_TEE_STATUS=0
    if [ "${#PIPESTATUS[@]}" -ge 1 ]; then
        RL_CMD_STATUS=${PIPESTATUS[0]}
    fi
    if [ "${#PIPESTATUS[@]}" -ge 2 ]; then
        RL_TEE_STATUS=${PIPESTATUS[1]}
    fi
    set -o pipefail
    if [ "$RL_CMD_STATUS" -ne 0 ]; then
        exit "$RL_CMD_STATUS"
    fi
    if [ "$RL_TEE_STATUS" -ne 0 ]; then
        echo "Warning: RL logging stream failed (tee exit ${RL_TEE_STATUS})." >&2
    fi

    FINAL_RL_ADAPTER="${RUN_DIR}/stage4/stage2_adapter.safetensors"
    if [ ! -f "$FINAL_RL_ADAPTER" ]; then
        echo "Error: Stage-2 RL failed to produce adapter weights at $FINAL_RL_ADAPTER"
        exit 1
    fi
else
    echo ""
    echo "[4/4] Skipping Stage-2 RL (training.stage2.enabled is false; set RUN_RL=1 to force)."
fi

echo ""
echo "=========================================================="
echo " Curriculum pipeline complete (SFT phases 1–3)."
echo " Final SFT adapter (phase 3): $PHASE3_ADAPTER"
if [ -n "$FINAL_RL_ADAPTER" ]; then
    echo " Final RL adapter (stage 2):  $FINAL_RL_ADAPTER"
fi
echo "=========================================================="
