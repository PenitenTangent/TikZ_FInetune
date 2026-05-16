#!/usr/bin/env python3
"""Rewrite a TikZ JSONL dataset to the tikz_body_only_v3 contract.

Usage:
  python3 tools/rewrite_prompt_contract.py --input data/prepared/val_unified_clean.jsonl --output data/prepared/val_unified_clean_v3.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from tikz_mlx.prompting import build_generation_prompt, PROMPT_CONTRACT_VERSION, prompt_template_sha256
from tikz_mlx.normalize import normalize_for_training_target


def _extract_description_and_hints(record: dict) -> tuple[str, str]:
    """Try to recover description and hints from the old prompt or record."""
    description = record.get("description", "")
    hints = record.get("hints", "")

    # Also check metadata for generation_mode
    meta_mode = record.get("metadata", {}).get("generation_mode")

    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if p.get("type") == "text")

            # If metadata mode is missing, try to find it in the prompt
            if not meta_mode:
                mode_match = re.search(r"mode:\s*(\w+)", content)
                if mode_match:
                    meta_mode = mode_match.group(1)

            # Strip old instruction boilerplate
            # Old contract v1/v2 patterns
            content = content.replace("Generate only the TikZ environment body according to the following requirements:", "")
            content = content.replace("Output constraints:", "")
            content = content.replace("- Generate only the TikZ environment body (e.g., \\begin{tikzpicture} ... \\end{tikzpicture}).", "")
            content = content.replace("- Do not output a LaTeX preamble, \\documentclass, \\usepackage, or \\PreviewEnvironment.", "")
            content = content.replace("- Do not output \\begin{document} or \\end{document}.", "")
            content = content.replace("- Start directly with the TikZ environment and end with the matching close and the markdown fence.", "")
            content = content.replace("- Preserve geometric constraints from the description (coordinates, labels, and relative placement).", "")
            content = content.replace("- Use strict TikZ syntax: terminate paths with ';', use calc ($...$) for math.", "")
            content = content.replace("```latex", "")

            # Also strip the GEOMETRY HINTS block if we found it
            content = re.sub(r"\[GEOMETRY HINTS\].*?```latex", "", content, flags=re.DOTALL)
            content = re.sub(r"\[GEOMETRY HINTS\].*$", "", content, flags=re.DOTALL)

            # Clean up double newlines
            while "\n\n\n" in content:
                content = content.replace("\n\n\n", "\n\n")

            if not description:
                description = content.strip()

    return description or "Unknown TikZ figure", meta_mode or "plain_tikz"


def rewrite_record(record: dict) -> dict:
    description, hints = _extract_description_and_hints(record)
    
    # Find assistant message and normalize target
    assistant_text = ""
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
            assistant_text = content
            break
            
    if not assistant_text:
        return record # Should not happen in clean data
        
    # Re-normalize to body-only
    # This strips \documentclass, \usepackage, \begin{document} and any trailing ```
    clean_target = normalize_for_training_target(assistant_text)
    
    # Ensure assistant target ends with exactly one ``` fence
    # normalize_for_training_target returns the LaTeX code without fences.
    # The new contract expects the assistant message to be: <latex>\n```
    target_with_fence = f"{clean_target}\n```"
    
    # Rebuild messages using the official template
    user_prompt = build_generation_prompt(description)
    
    # The assistant message in the dataset should be the full target including fences
    # The template ends with ```latex, so the assistant starts with the code
    # Wait, the template ends with ```latex\n.
    # Actually, build_generation_prompt returns exactly what is shown in viewed_file.
    
    new_messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": target_with_fence}
    ]

    # Preserve old metadata but override contract fields
    new_metadata = dict(record.get("metadata", {}))
    new_metadata["prompt_contract_version"] = PROMPT_CONTRACT_VERSION
    new_metadata["prompt_template_sha256"] = prompt_template_sha256()
    new_metadata["target_contract"] = "body_only_environment"

    # Ensure generation_mode is set correctly if we recovered it
    if hints: # hints is actually meta_mode in this script's return signature
        new_metadata["generation_mode"] = hints

    new_record = {
        "sample_id": record.get("sample_id"),
        "messages": new_messages,
        "metadata": new_metadata
    }
    
    # Preserve any other useful fields
    for k in ["group_id", "complexity", "source"]:
        if k in record:
            new_record[k] = record[k]
            
    return new_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate JSONL to v3 contract.")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not in_path.exists():
        print(f"ERROR: {in_path} not found.")
        sys.exit(1)
        
    print(f"Migrating {in_path} -> {out_path} ...")
    
    with in_path.open("r", encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            new_rec = rewrite_record(rec)
            f_out.write(json.dumps(new_rec) + "\n")
            
    print("Migration complete.")


if __name__ == "__main__":
    main()
