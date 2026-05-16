#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.validate_split_integrity import (
    assistant_text,
    load_jsonl,
    normalized_prompt_hash,
    normalized_target_hash,
    prompt_text,
    sample_id,
    shingles,
    source_key,
    validate_ngram_contamination,
    write_integrity_manifest,
)


def _is_safe_candidate(
    candidate: dict,
    candidate_index: int,
    *,
    train_records: list[dict],
    selected_records: list[dict],
    used_sources: set[str],
    used_prompt_hashes: set[str],
    used_target_hashes: set[str],
    code_ngram_size: int,
    prompt_ngram_size: int,
    max_code_containment: float,
    max_prompt_containment: float,
) -> tuple[bool, str | None]:
    sid = sample_id(candidate, candidate_index)
    source = source_key(candidate)
    if source is not None and source in used_sources:
        return False, f"{sid}: source reused: {source}"

    prompt_hash = normalized_prompt_hash(candidate) if prompt_text(candidate).strip() else None
    target_hash = normalized_target_hash(candidate) if assistant_text(candidate).strip() else None
    if prompt_hash is not None and prompt_hash in used_prompt_hashes:
        return False, f"{sid}: prompt hash reused"
    if target_hash is not None and target_hash in used_target_hashes:
        return False, f"{sid}: target hash reused"

    violations = validate_ngram_contamination(
        {"train": train_records + selected_records, "candidate": [candidate]},
        code_shingle_size=code_ngram_size,
        prompt_shingle_size=prompt_ngram_size,
        max_code_containment=max_code_containment,
        max_prompt_containment=max_prompt_containment,
    )
    if violations:
        return False, f"{sid}: {violations[0].message}"
    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a contamination-safe eval split from candidate JSONL.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Stop after selecting this many examples; 0 means all safe examples.")
    parser.add_argument("--code-ngram-size", type=int, default=13)
    parser.add_argument("--prompt-ngram-size", type=int, default=8)
    parser.add_argument("--max-code-containment", type=float, default=0.20)
    parser.add_argument("--max-prompt-containment", type=float, default=0.50)
    args = parser.parse_args()

    train_records = load_jsonl(Path(args.train))
    candidate_records = load_jsonl(Path(args.candidates))
    selected: list[dict] = []
    rejected: list[str] = []
    used_sources = {source for record in train_records if (source := source_key(record)) is not None}
    used_prompt_hashes = {
        normalized_prompt_hash(record)
        for record in train_records
        if prompt_text(record).strip()
    }
    used_target_hashes = {
        normalized_target_hash(record)
        for record in train_records
        if assistant_text(record).strip()
    }

    for index, candidate in enumerate(candidate_records):
        safe, reason = _is_safe_candidate(
            candidate,
            index,
            train_records=train_records,
            selected_records=selected,
            used_sources=used_sources,
            used_prompt_hashes=used_prompt_hashes,
            used_target_hashes=used_target_hashes,
            code_ngram_size=args.code_ngram_size,
            prompt_ngram_size=args.prompt_ngram_size,
            max_code_containment=args.max_code_containment,
            max_prompt_containment=args.max_prompt_containment,
        )
        if not safe:
            rejected.append(reason or f"{sample_id(candidate, index)}: rejected")
            continue
        selected.append(candidate)
        source = source_key(candidate)
        if source is not None:
            used_sources.add(source)
        if prompt_text(candidate).strip():
            used_prompt_hashes.add(normalized_prompt_hash(candidate))
        if assistant_text(candidate).strip():
            used_target_hashes.add(normalized_target_hash(candidate))
        if args.limit > 0 and len(selected) >= args.limit:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    write_integrity_manifest(
        Path(args.manifest_out),
        {
            "pass": True,
            "train_count": len(train_records),
            "candidate_count": len(candidate_records),
            "selected_count": len(selected),
            "rejected_count": len(rejected),
            "selected_sample_ids": [sample_id(record, i) for i, record in enumerate(selected)],
            "rejections": rejected[:200],
            "settings": {
                "code_ngram_size": args.code_ngram_size,
                "prompt_ngram_size": args.prompt_ngram_size,
                "max_code_containment": args.max_code_containment,
                "max_prompt_containment": args.max_prompt_containment,
            },
        },
    )
    print(f"Wrote contamination-safe eval split: {output} ({len(selected)} selected)")


if __name__ == "__main__":
    main()
