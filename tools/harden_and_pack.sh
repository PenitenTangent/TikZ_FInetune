#!/bin/bash
# harden_and_pack.sh — End-to-end dataset preparation pipeline.
#
# Stages:
#   1. Harden  — static critic, description quality, compile-and-repair,
#                bounding-box check, perceptual dedup, pedagogical scoring,
#                curriculum sort.
#   2. Tokenize — convert hardened JSONL to raw token arrays (.npy).
#   3. Pack     — pack into fixed-length buffers with masks, boundaries,
#                 and optional syntax weights.
#
# Usage:
#   ./tools/harden_and_pack.sh <config_yaml> <input_jsonl> <output_prefix> [cache_db]
#
# Example:
#   ./tools/harden_and_pack.sh \
#       configs/run_300_iters.yaml \
#       data/prepared/train.jsonl \
#       data/prepared/train_final \
#       data/cache/compile_cache.db
#
# On second run, pass the same cache_db to skip recompiling records that
# already have a cached result (saves hours of Tectonic time).

set -euo pipefail

CONFIG=${1:?Usage: $0 <config_yaml> <input_jsonl> <output_prefix> [cache_db]}
INPUT=${2:?Usage: $0 <config_yaml> <input_jsonl> <output_prefix> [cache_db]}
OUTPUT_PREFIX=${3:?Usage: $0 <config_yaml> <input_jsonl> <output_prefix> [cache_db]}
CACHE_DB=${4:-}  # optional

HARDENED_JSONL="${OUTPUT_PREFIX}_hardened.jsonl"
TOKENIZED_NPY="${OUTPUT_PREFIX}_tokenized.npy"
AUDIT_JSON="${OUTPUT_PREFIX}_tokenize_audit.json"
PACKED_NPY="${OUTPUT_PREFIX}_packed.npy"

# Derive model and assistant IDs from config
MODEL_ID=$(.venv/bin/python -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg['model']['model_id'])
")
ASSISTANT_ID=$(.venv/bin/python -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg['training'].get('assistant_id', 4368))
")

echo ""
echo "=========================================="
echo " STAGE 1 — HARDENING"
echo "=========================================="
echo "  Input:  $INPUT"
echo "  Output: $HARDENED_JSONL"
[ -n "$CACHE_DB" ] && echo "  Cache:  $CACHE_DB (reusing compiled results)"
echo ""

HARDEN_ARGS=(
    --config  "$CONFIG"
    --input   "$INPUT"
    --output  "$HARDENED_JSONL"
    --workers 11
)
[ -n "$CACHE_DB" ] && HARDEN_ARGS+=(--cache "$CACHE_DB")

.venv/bin/python -m tikz_mlx.cli harden-dataset "${HARDEN_ARGS[@]}"

echo ""
echo "=========================================="
echo " STAGE 2 — PRE-TOKENIZING"
echo "=========================================="
echo "  Input:  $HARDENED_JSONL"
echo "  Output: $TOKENIZED_NPY"
echo "  Model:  $MODEL_ID"
echo ""

.venv/bin/python tools/pretokenize_dataset.py \
    --model-id "$MODEL_ID" \
    --dataset  "$HARDENED_JSONL" \
    --output   "$TOKENIZED_NPY"

echo ""
echo "=========================================="
echo " STAGE 3 — PACKING"
echo "=========================================="
echo "  Input:  $TOKENIZED_NPY"
echo "  Output: $PACKED_NPY"
echo "  Asst token ID: $ASSISTANT_ID"
echo ""

.venv/bin/python tools/pack_tokenized_dataset.py \
    --input           "$TOKENIZED_NPY" \
    --output          "$PACKED_NPY" \
    --assistant-token "$ASSISTANT_ID" \
    --metadata-jsonl  "$HARDENED_JSONL" \
    --emit-syntax-weights \
    --model-id        "$MODEL_ID"

echo ""
echo "=========================================="
echo " COMPLETE"
echo "=========================================="
echo ""
echo "Packed data:    $PACKED_NPY"
echo "Masks:          ${OUTPUT_PREFIX}_packed_masks.npy"
echo "Boundaries:     ${OUTPUT_PREFIX}_packed_boundaries.npy"
echo "Syntax weights: ${OUTPUT_PREFIX}_packed_syntax_weights.npy"
echo ""
echo "Update your training config:"
echo "  pretokenized_packed_cache_path: $PACKED_NPY"
echo "  repair_before_training: false"
echo "  static_critic_training_gate: false"
echo ""
echo "To rerun with the compiler cache, use:"
echo "  $0 $CONFIG $INPUT $OUTPUT_PREFIX ${CACHE_DB:-data/cache/compile_cache.db}"
