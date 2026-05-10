#!/usr/bin/env python3
"""Audit a training JSONL for prompt-contract compliance.

Checks:
  USER prompt:
    - No actual preamble injection: \\PreviewEnvironment{ (with brace = actual usage)
    - No \\usepackage[active,tightpage]{preview}
    - No "--- Starting Preamble ---"
    (negation instructions like "Do not output \\documentclass" are ALLOWED)

  ASSISTANT target:
    - No \\documentclass
    - No \\usepackage (any form)
    - No \\PreviewEnvironment
    - No \\begin{document}
    - No \\end{document}
    - Must start with a known TikZ environment open
    - Must end with exactly one closing markdown fence (```)

Usage:
  python3 tools/audit_prompt_contract.py --input data/prepared/curriculum/train_stage1_clean.jsonl
  python3 tools/audit_prompt_contract.py --input data/prepared/curriculum/train_stage1_clean.jsonl --fail
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Patterns that indicate ACTUAL USAGE in the user prompt (not mere mention).
# We use regexes to catch usage like \documentclass{ but ignore "Do not output \documentclass"
PROMPT_FORBIDDEN_USAGE_RE = [
    re.compile(r"\\documentclass(?:\[[^\]]*\])?\{"),
    re.compile(r"\\usepackage(?:\[[^\]]*\])?\{"),
    re.compile(r"\\usetikzlibrary\{"),
    re.compile(r"\\begin\{document\}"),
    re.compile(r"\\end\{document\}"),
    re.compile(r"\\PreviewEnvironment\{"),
    re.compile(r"--- Starting Preamble ---"),
]

# Patterns that must NOT appear in the assistant target (even as partial strings)
ASSISTANT_FORBIDDEN = [
    "\\documentclass",
    "\\usepackage",
    "\\PreviewEnvironment",
    "\\begin{document}",
    "\\end{document}",
]

# Assistant target must start with one of these environments
VALID_ENV_STARTS = re.compile(
    r"^\s*\\begin\{(?:tikzpicture|tikz-cd|circuitikz|axis|tikzpicture\*)\}",
    re.IGNORECASE,
)

# Count backtick fence tokens (3 consecutive backticks, not 6)
FENCE_RE = re.compile(r"(?<![`])```(?![`])")


def _extract_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content or ""


def audit_file(path: Path) -> dict:
    counts: dict = {
        "record_count": 0,
        "prompt_violations": {},
        "assistant_violations": {},
        "assistant_no_env_start": 0,
        "assistant_fence_not_exactly_once": 0,
        "violation_sample_ids": [],
    }

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            counts["record_count"] += 1
            sample_id = str(record.get("sample_id", f"row_{counts['record_count']}"))

            # Check metadata
            metadata = record.get("metadata", {})
            if metadata.get("prompt_contract_version") != "tikz_body_only_v3":
                counts["violation_sample_ids"].append(sample_id)
                counts["total_violations"] += 1
                continue

            messages = record.get("messages", [])
            record_has_violation = False

            for msg in messages:
                role = msg.get("role", "")
                text = _extract_text(msg)

                if role == "user":
                    for rex in PROMPT_FORBIDDEN_USAGE_RE:
                        if rex.search(text):
                            pat = rex.pattern
                            counts["prompt_violations"][pat] = (
                                counts["prompt_violations"].get(pat, 0) + 1
                            )
                            record_has_violation = True

                elif role == "assistant":
                    for token in ASSISTANT_FORBIDDEN:
                        if token in text:
                            counts["assistant_violations"][token] = (
                                counts["assistant_violations"].get(token, 0) + 1
                            )
                            record_has_violation = True

                    if not VALID_ENV_STARTS.match(text):
                        counts["assistant_no_env_start"] += 1
                        record_has_violation = True

                    # The format is: <env body>\n``` — one fence at the very end
                    fence_count = len(FENCE_RE.findall(text))
                    if fence_count != 1:
                        counts["assistant_fence_not_exactly_once"] += 1
                        record_has_violation = True

            if record_has_violation:
                counts["violation_sample_ids"].append(sample_id)

    total_prompt_violations = sum(counts["prompt_violations"].values())
    total_assistant_violations = sum(counts["assistant_violations"].values())
    counts["total_violations"] = (
        total_prompt_violations
        + total_assistant_violations
        + counts["assistant_no_env_start"]
        + counts["assistant_fence_not_exactly_once"]
    )
    counts["violation_record_count"] = len(counts["violation_sample_ids"])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prompt contract compliance.")
    parser.add_argument("--input", required=True, help="JSONL file to audit")
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Exit 1 if any violations are found",
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=0,
        help="Maximum allowed violations before failing (default: 0)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Auditing {path} ...", flush=True)
    result = audit_file(path)

    print(json.dumps(result, indent=2))

    total = result["total_violations"]
    if total > 0:
        print(f"\n❌ {total} violation(s) in {result['violation_record_count']} record(s).", flush=True)
        if result["violation_sample_ids"][:10]:
            print("First failing sample IDs:", result["violation_sample_ids"][:10])
    else:
        print(f"\n✅ All {result['record_count']} records pass contract audit.", flush=True)

    if args.fail and total > args.max_violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
