#!/usr/bin/env bash

resume_offset_from_checkpoint_path() {
  local checkpoint_path="$1"
  local checkpoint_filename
  checkpoint_filename=$(basename "$checkpoint_path")

  if [[ "$checkpoint_filename" =~ ^([0-9]+)_adapters\.safetensors$ ]]; then
    local resume_offset="${BASH_REMATCH[1]}"
    echo "$((10#$resume_offset))"
  else
    echo "0"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [ "$#" -ne 1 ]; then
    echo "Usage: resume_offset.sh <checkpoint-path>" >&2
    exit 2
  fi
  resume_offset_from_checkpoint_path "$1"
fi
