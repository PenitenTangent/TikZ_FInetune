#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.validate_split_integrity import (
    assistant_text,
    contamination_prompt_text,
    load_jsonl,
    normalized_prompt_hash,
    normalized_target_hash,
    sample_id,
    shingles,
)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            row = dict(record)
            metadata = dict(row.get("metadata") or {})
            row["example_index"] = index
            metadata["example_index"] = index
            row["metadata"] = metadata
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _exact_contaminated_train_ids(train: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> set[str]:
    eval_target_hashes = {
        normalized_target_hash(record)
        for record in eval_records
        if assistant_text(record).strip()
    }
    eval_prompt_hashes = {
        normalized_prompt_hash(record)
        for record in eval_records
        if contamination_prompt_text(record).strip()
    }
    contaminated: set[str] = set()
    for index, record in enumerate(train):
        sid = sample_id(record, index)
        if assistant_text(record).strip() and normalized_target_hash(record) in eval_target_hashes:
            contaminated.add(sid)
        if contamination_prompt_text(record).strip() and normalized_prompt_hash(record) in eval_prompt_hashes:
            contaminated.add(sid)
    return contaminated


def _ngram_contaminated_train_ids(
    train: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    code_ngram_size: int,
    prompt_ngram_size: int,
    max_code_containment: float,
    max_prompt_containment: float,
) -> set[str]:
    def build_index(extractor, shingle_size: int) -> dict[tuple[str, ...], list[str]]:
        index: dict[tuple[str, ...], list[str]] = {}
        for row_index, row in enumerate(train):
            sid = sample_id(row, row_index)
            for shingle in shingles(extractor(row), shingle_size):
                index.setdefault(shingle, []).append(sid)
        return index

    def collect(eval_shingles: set[tuple[str, ...]], index: dict[tuple[str, ...], list[str]], threshold: float) -> set[str]:
        if not eval_shingles:
            return set()
        counts: dict[str, int] = {}
        for shingle in eval_shingles:
            for train_id in index.get(shingle, []):
                counts[train_id] = counts.get(train_id, 0) + 1
        return {
            train_id
            for train_id, overlap_count in counts.items()
            if overlap_count / len(eval_shingles) > threshold
        }

    code_index = build_index(assistant_text, code_ngram_size)
    prompt_index = build_index(contamination_prompt_text, prompt_ngram_size)
    contaminated: set[str] = set()
    for eval_record in eval_records:
        contaminated.update(
            collect(shingles(assistant_text(eval_record), code_ngram_size), code_index, max_code_containment)
        )
        contaminated.update(
            collect(
                shingles(contamination_prompt_text(eval_record), prompt_ngram_size),
                prompt_index,
                max_prompt_containment,
            )
        )
    return contaminated


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop training rows that contaminate validation/eval splits.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", action="append", default=[], help="Validation/eval JSONL path; may be repeated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--code-ngram-size", type=int, default=13)
    parser.add_argument("--prompt-ngram-size", type=int, default=8)
    parser.add_argument("--max-code-containment", type=float, default=0.20)
    parser.add_argument("--max-prompt-containment", type=float, default=0.50)
    parser.add_argument("--max-passes", type=int, default=50)
    args = parser.parse_args()

    train = load_jsonl(Path(args.train))
    original_count = len(train)
    eval_records: list[dict[str, Any]] = []
    for eval_path in args.eval:
        path = Path(eval_path)
        if path.exists():
            eval_records.extend(load_jsonl(path))

    dropped_ids: list[str] = []
    for _ in range(max(1, args.max_passes)):
        contaminated = _exact_contaminated_train_ids(train, eval_records)
        contaminated.update(
            _ngram_contaminated_train_ids(
                train,
                eval_records,
                code_ngram_size=args.code_ngram_size,
                prompt_ngram_size=args.prompt_ngram_size,
                max_code_containment=args.max_code_containment,
                max_prompt_containment=args.max_prompt_containment,
            )
        )
        if not contaminated:
            break
        dropped_ids.extend(sorted(contaminated))
        train = [
            record
            for index, record in enumerate(train)
            if sample_id(record, index) not in contaminated
        ]

    _write_jsonl(Path(args.output), train)
    audit = {
        "input_records": original_count,
        "output_records": len(train),
        "dropped_records": len(dropped_ids),
        "dropped_sample_ids": dropped_ids[:500],
        "settings": {
            "code_ngram_size": args.code_ngram_size,
            "prompt_ngram_size": args.prompt_ngram_size,
            "max_code_containment": args.max_code_containment,
            "max_prompt_containment": args.max_prompt_containment,
            "max_passes": args.max_passes,
        },
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Decontaminated training set: dropped {len(dropped_ids)} rows, kept {len(train)} rows.")


if __name__ == "__main__":
    main()
