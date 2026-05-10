#!/bin/bash
# run_data_gates.sh — mandatory data quality gates before training.
#
# Usage:
#   ./tools/run_data_gates.sh --stage 1 --input data/prepared/curriculum/train_stage1.jsonl
#   ./tools/run_data_gates.sh --stage 1 --input data/prepared/curriculum/train_stage1.jsonl \
#       --val data/prepared/val.jsonl --gold data/prepared/gold_eval.jsonl
#
# What it does (in order, all abort on failure):
#   0.  audit_prompt_contract    — hard-fail if gold eval JSONL is poisoned
#   1.  filter_training_records  — drop bad-pattern / low-substantive records
#   1.5 audit_prompt_contract    — hard-fail if clean JSONL is poisoned
#   2.  audit_training_records   — compute stats on raw and clean datasets
#   3.  diff_dataset_audits      — compare raw vs clean, fail on warnings
#   4.  validate_split_integrity — ensure no ID / target-hash leakage
#   5.  pretokenize_dataset      — tokenize with max_tokens from stage config
#
# Outputs are written to data/prepared/curriculum/gates/<stage>/ by default,
# or to --clean-output and --pretok-output paths if provided.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  PYTHON_EXE=".venv/bin/python"
else
  PYTHON_EXE="${PYTHON_EXE:-python3}"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ── Argument parsing ────────────────────────────────────────────────────────
STAGE_NUM=""
INPUT_JSONL=""
VAL_JSONL=""
GOLD_JSONL=""
CLEAN_JSONL_OVERRIDE=""
PRETOK_OUT_OVERRIDE=""
SKIP_PRETOKENIZE="0"

while [ $# -gt 0 ]; do
  case "$1" in
    --stage)    STAGE_NUM="${2:?--stage requires a number}"; shift 2 ;;
    --input)    INPUT_JSONL="${2:?--input requires a path}"; shift 2 ;;
    --val)      VAL_JSONL="${2:?--val requires a path}";   shift 2 ;;
    --gold)     GOLD_JSONL="${2:?--gold requires a path}"; shift 2 ;;
    --clean-output)  CLEAN_JSONL_OVERRIDE="${2:?--clean-output requires a path}"; shift 2 ;;
    --pretok-output) PRETOK_OUT_OVERRIDE="${2:?--pretok-output requires a path}"; shift 2 ;;
    --skip-pretokenize) SKIP_PRETOKENIZE="1"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$STAGE_NUM" ] || [ -z "$INPUT_JSONL" ]; then
  echo "Usage: run_data_gates.sh --stage <1-5> --input <train.jsonl> [--val <val.jsonl>] [--gold <gold.jsonl>]" >&2
  exit 1
fi

if ! [[ "$STAGE_NUM" =~ ^[1-5]$ ]]; then
  echo "ERROR: Stage must be 1-5, got: $STAGE_NUM" >&2; exit 1
fi

CONFIG="configs/curriculum_stage${STAGE_NUM}.yaml"
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: Stage config not found: $CONFIG" >&2; exit 1
fi

GATE_DIR="data/prepared/curriculum/gates/stage${STAGE_NUM}"
mkdir -p "$GATE_DIR"

CLEAN_JSONL="${CLEAN_JSONL_OVERRIDE:-${GATE_DIR}/train_stage${STAGE_NUM}_clean.jsonl}"
REJECTED_JSONL="${GATE_DIR}/train_stage${STAGE_NUM}_rejected.jsonl"
RAW_AUDIT="${GATE_DIR}/audit_raw.json"
CLEAN_AUDIT="${GATE_DIR}/audit_clean.json"
DIFF_OUT="${GATE_DIR}/diff_raw_vs_clean.json"
INTEGRITY_OUT="${GATE_DIR}/split_integrity.json"
PRETOK_OUT="${PRETOK_OUT_OVERRIDE:-${GATE_DIR}/train_stage${STAGE_NUM}_clean_tokenized.npy}"

echo ""
echo "========================================="
echo "  Data gates — Stage $STAGE_NUM"
echo "========================================="
echo "  Input:   $INPUT_JSONL"
echo "  Config:  $CONFIG"
echo "  Gates:   $GATE_DIR"
echo ""

# ── Gate 0: Contract audit on gold eval ─────────────────────────────────────
if [ -n "$GOLD_JSONL" ] && [ -f "$GOLD_JSONL" ]; then
  echo ">>> [0/5] Auditing gold eval prompt contract..."
  "$PYTHON_EXE" tools/audit_prompt_contract.py \
    --input "$GOLD_JSONL" \
    --fail
  echo "    ✓ Gold eval contract audit passed"
  echo ""
fi

# ── Gate 1: Filter ──────────────────────────────────────────────────────────
echo ">>> [1/5] Filtering training records..."
"$PYTHON_EXE" tools/filter_training_records.py \
  --input  "$INPUT_JSONL" \
  --output "$CLEAN_JSONL" \
  --rejected "$REJECTED_JSONL"
echo "    ✓ Filter complete → $CLEAN_JSONL"

# ── Gate 1.5: Contract audit on clean JSONL ──────────────────────────────────
echo ""
echo ">>> [1.5/5] Auditing clean JSONL prompt contract..."
"$PYTHON_EXE" tools/audit_prompt_contract.py \
  --input "$CLEAN_JSONL" \
  --fail
echo "    ✓ Clean JSONL contract audit passed"

# ── Gate 2: Audit (raw + clean) ─────────────────────────────────────────────
echo ""
echo ">>> [2/5] Auditing datasets..."
"$PYTHON_EXE" tools/audit_training_records.py \
  --input "$INPUT_JSONL" \
  --out   "$RAW_AUDIT"
"$PYTHON_EXE" tools/audit_training_records.py \
  --input "$CLEAN_JSONL" \
  --out   "$CLEAN_AUDIT"
echo "    ✓ Audit complete → $RAW_AUDIT, $CLEAN_AUDIT"

# ── Gate 3: Diff (fail on warnings) ─────────────────────────────────────────
echo ""
echo ">>> [3/5] Diffing raw vs clean (fail-on-warning)..."
"$PYTHON_EXE" tools/diff_dataset_audits.py \
  --raw   "$RAW_AUDIT" \
  --clean "$CLEAN_AUDIT" \
  --out   "$DIFF_OUT" \
  --fail-on-warning
echo "    ✓ Diff gate passed → $DIFF_OUT"

# ── Gate 4: Split integrity ─────────────────────────────────────────────────
echo ""
echo ">>> [4/5] Validating split integrity..."
SPLIT_ARGS="--split train=${CLEAN_JSONL}"
if [ -n "$VAL_JSONL" ] && [ -f "$VAL_JSONL" ]; then
  SPLIT_ARGS="$SPLIT_ARGS --split val=${VAL_JSONL}"
fi
if [ -n "$GOLD_JSONL" ] && [ -f "$GOLD_JSONL" ]; then
  SPLIT_ARGS="$SPLIT_ARGS --split gold_eval=${GOLD_JSONL}"
fi
# shellcheck disable=SC2086
"$PYTHON_EXE" tools/validate_split_integrity.py \
  $SPLIT_ARGS \
  --out "$INTEGRITY_OUT"
echo "    ✓ Split integrity passed → $INTEGRITY_OUT"

# ── Gate 5: Pre-tokenize ─────────────────────────────────────────────────────
if [ "$SKIP_PRETOKENIZE" = "1" ]; then
  echo ""
  echo ">>> [5/5] Pretokenization skipped (--skip-pretokenize)."
else
  echo ""
  echo ">>> [5/5] Pre-tokenizing (max_tokens from $CONFIG)..."
  "$PYTHON_EXE" tools/pretokenize_dataset.py \
    --config  "$CONFIG" \
    --dataset "$CLEAN_JSONL" \
    --output  "$PRETOK_OUT"
  echo "    ✓ Pretokenization complete → $PRETOK_OUT"
fi

echo ""
echo "========================================="
echo "  All data gates passed for Stage $STAGE_NUM"
echo "========================================="
echo "  Clean dataset: $CLEAN_JSONL"
[ "$SKIP_PRETOKENIZE" = "0" ] && echo "  Tokenized:     $PRETOK_OUT"
echo ""
