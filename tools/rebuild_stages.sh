#!/bin/bash
set -euo pipefail

STAGES=(1 2 3 4 5)
MAX_TOKENS=(768 1024 1536 1792 1280)

for i in "${!STAGES[@]}"; do
  stage="${STAGES[$i]}"
  max_tok="${MAX_TOKENS[$i]}"
  
  echo "Processing Stage $stage with max_tokens $max_tok..."
  
  python3 tools/recovery_artifacts.py filter-clean-data \
    --input "data/prepared/curriculum/train_stage${stage}.jsonl" \
    --output "data/prepared/curriculum/train_stage${stage}_clean.jsonl" \
    --audit-output "data/prepared/curriculum/train_stage${stage}_clean_audit.json" \
    --max-token-length "$max_tok" \
    --repair-contract
    
  python3 tools/pretokenize_dataset.py \
    --model-id "mlx-community/gemma-4-e4b-it-6bit" \
    --dataset "data/prepared/curriculum/train_stage${stage}_clean.jsonl" \
    --output "data/prepared/curriculum/train_stage${stage}_clean_tokenized.npy" \
    --max-tokens "$max_tok" \
    --prompt-contract-version "tikz_partial_decode_v1" \
    --disabled-rules ""
done

echo "Done rebuilding stages."
