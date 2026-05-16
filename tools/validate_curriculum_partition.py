#!/usr/bin/env python3
import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import Set

def get_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Validate that curriculum stages are disjoint.")
    parser.add_argument("--stage", action="append", required=True, help="Path to a stage JSONL file.")
    args = parser.parse_args()

    sample_ids: dict[str, str] = {} # id -> stage_path
    target_hashes: dict[str, str] = {} # hash -> stage_path
    content_hashes: dict[str, str] = {} # hash -> stage_path

    overlaps = []

    for stage_path in args.stage:
        path = Path(stage_path)
        if not path.exists():
            print(f"ERROR: Stage file not found: {stage_path}")
            sys.exit(1)
        
        print(f"Auditing stage: {path.name}...")
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"ERROR: Invalid JSON in {stage_path} at line {line_idx}")
                    continue

                sample_id = record.get("sample_id")
                messages = record.get("messages", [])
                def _get_text(c):
                    if isinstance(c, str):
                        return c
                    if isinstance(c, list):
                        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
                    return ""

                assistant_text = _get_text(messages[1].get("content", "")) if len(messages) >= 2 and messages[1].get("role") == "assistant" else ""
                prompt_text = _get_text(messages[0].get("content", "")) if messages else ""
                full_content = f"{prompt_text}\n{assistant_text}"
                c_hash = get_content_hash(full_content)
                t_hash = get_content_hash(assistant_text) if assistant_text else ""

                if sample_id:
                    if sample_id in sample_ids:
                        prev_stage = sample_ids[sample_id]
                        if prev_stage != stage_path:
                            overlaps.append(f"Inter-stage duplicate sample_id '{sample_id}' in {path.name} (also in {Path(prev_stage).name})")
                    else:
                        sample_ids[sample_id] = stage_path

                if t_hash:
                    if t_hash in target_hashes:
                        prev_stage = target_hashes[t_hash]
                        if prev_stage != stage_path:
                            overlaps.append(f"Inter-stage duplicate assistant target hash in {path.name} (also in {Path(prev_stage).name})")
                    else:
                        target_hashes[t_hash] = stage_path

                if c_hash:
                    if c_hash in content_hashes:
                        prev_stage = content_hashes[c_hash]
                        if prev_stage != stage_path:
                            overlaps.append(f"Inter-stage duplicate content hash in {path.name} (also in {Path(prev_stage).name})")
                    else:
                        content_hashes[c_hash] = stage_path

    if overlaps:
        print("\nERROR: Found inter-stage overlaps (partition violated):")
        for overlap in overlaps[:20]:
            print(f"  - {overlap}")
        if len(overlaps) > 20:
            print(f"  ... and {len(overlaps) - 20} more.")
        sys.exit(1)
    else:
        print("\nSUCCESS: All stages are disjoint (no inter-stage overlaps).")

if __name__ == "__main__":
    main()
