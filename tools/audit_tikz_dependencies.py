#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from tikz_mlx.normalize import contains_external_dependencies, detect_required_packages


USEPACKAGE_RE = re.compile(r"\\usepackage(?:\[[^\]]+\])?\{([^}]+)\}")
USETIKZLIBRARY_RE = re.compile(r"\\usetikzlibrary\{([^}]+)\}")


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def extract_tikz_text(record: dict[str, Any]) -> str:
    for key in ("reference_code", "normalized_code", "raw_code", "tikz", "code", "assistant"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                text = _extract_text_from_content(msg.get("content"))
                if text.strip():
                    return text
    return ""


def _split_tex_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _dependency_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    package_counts: Counter[str] = Counter()
    library_counts: Counter[str] = Counter()
    external_dependency_hints: list[dict[str, str]] = []
    empty_records = 0
    for index, record in enumerate(records):
        text = extract_tikz_text(record)
        if not text.strip():
            empty_records += 1
            continue
        if contains_external_dependencies(text):
            external_dependency_hints.append(
                {
                    "sample_id": str(record.get("sample_id", f"row_{index:06d}")),
                    "hint": "external dependency command detected",
                }
            )
        preamble = "\n".join(detect_required_packages(text))
        for match in USEPACKAGE_RE.finditer(preamble):
            package_counts.update(_split_tex_list(match.group(1)))
        for match in USETIKZLIBRARY_RE.finditer(preamble):
            library_counts.update(_split_tex_list(match.group(1)))
    return {
        "records": len(records),
        "empty_records": empty_records,
        "packages": dict(sorted(package_counts.items())),
        "tikz_libraries": dict(sorted(library_counts.items())),
        "external_dependency_hints": external_dependency_hints[:200],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dynamically detected TikZ packages/libraries.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    payload = _dependency_counts(records)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote dependency audit: {output}")


if __name__ == "__main__":
    main()
