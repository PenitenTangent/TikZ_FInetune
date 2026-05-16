import argparse
import json
import hashlib
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class Violation:
    type: str
    message: str
    ids: List[str]

def load_jsonl(path: Path) -> List[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def sample_id(record: dict, index: int) -> str:
    return str(record.get("sample_id") or f"row_{index:06d}")

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

def assistant_text(record: dict) -> str:
    for key in ("reference_code", "normalized_code", "raw_code", "tikz", "code", "assistant"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return _extract_text_from_content(msg.get("content", ""))
    return ""

def prompt_text(record: dict) -> str:
    for key in ("prompt_text", "description", "prompt", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return _extract_text_from_content(msg.get("content", ""))
    return ""

def contamination_prompt_text(record: dict) -> str:
    text = prompt_text(record)
    if not text:
        return ""
    text = re.sub(
        r"^Generate only the TikZ environment body according to the following requirements:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.split(r"\n\[GEOMETRY HINTS\]|\nOutput constraints:", text, maxsplit=1)[0]
    text = re.sub(r"```latex\s*$", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()

def normalize_text(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text

def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()

def normalized_target_hash(record: dict) -> str:
    return normalized_hash(assistant_text(record))

def normalized_prompt_hash(record: dict) -> str:
    return normalized_hash(contamination_prompt_text(record))

def text_tokens(text: str) -> list[str]:
    return re.findall(r"\\[A-Za-z]+|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", normalize_text(text))

def shingles(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = text_tokens(text)
    if size <= 0 or len(tokens) < size:
        return set()
    return {tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}

def source_key(record: dict) -> str | None:
    candidates = [
        record.get("source_id"),
        record.get("paper_id"),
        record.get("repo"),
        record.get("arxiv_id"),
    ]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("source"),
                metadata.get("source_id"),
                metadata.get("paper_id"),
                metadata.get("repo"),
                metadata.get("arxiv_id"),
                metadata.get("source_path"),
                metadata.get("path"),
            ]
        )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in (record.get("source"), metadata.get("source") if isinstance(metadata, dict) else None):
        if isinstance(value, str) and value.strip() and re.search(r"[/\\:]|arxiv|\.tex$|\.jsonl?$", value, flags=re.IGNORECASE):
            return value.strip()
    return None

def validate_no_duplicate_ids(split_name: str, records: List[dict]) -> List[Violation]:
    seen = set()
    dupes = set()
    for i, rec in enumerate(records):
        sid = sample_id(rec, i)
        if sid in seen:
            dupes.add(sid)
        seen.add(sid)
    if dupes:
        return [Violation("duplicate_ids_in_split", f"Duplicate IDs found in {split_name}", list(dupes))]
    return []

def validate_no_cross_split_id_overlap(splits: Dict[str, List[dict]]) -> List[Violation]:
    violations = []
    split_names = list(splits.keys())
    pairs = (
        [("train", name) for name in split_names if name != "train"]
        if "train" in splits
        else [(split_names[i], split_names[j]) for i in range(len(split_names)) for j in range(i + 1, len(split_names))]
    )
    
    for name1, name2 in pairs:
        ids1 = {sample_id(r, j) for j, r in enumerate(splits[name1])}
        ids2 = {sample_id(r, k) for k, r in enumerate(splits[name2])}

        overlap = ids1.intersection(ids2)
        if overlap:
            violations.append(Violation(
                "cross_split_id_overlap",
                f"Overlap between {name1} and {name2}",
                list(overlap)
            ))
    return violations

def validate_no_cross_split_target_hash_overlap(splits: Dict[str, List[dict]]) -> List[Violation]:
    violations = []
    split_names = list(splits.keys())
    pairs = (
        [("train", name) for name in split_names if name != "train"]
        if "train" in splits
        else [(split_names[i], split_names[j]) for i in range(len(split_names)) for j in range(i + 1, len(split_names))]
    )
    
    for name1, name2 in pairs:
        hashes1 = {normalized_target_hash(r) for r in splits[name1] if assistant_text(r).strip()}
        hashes2 = {normalized_target_hash(r) for r in splits[name2] if assistant_text(r).strip()}

        overlap = hashes1.intersection(hashes2)
        if overlap:
            violations.append(Violation(
                "cross_split_target_hash_overlap",
                f"Target text overlap between {name1} and {name2}",
                list(overlap)
            ))
    return violations

def validate_no_cross_split_prompt_hash_overlap(splits: Dict[str, List[dict]]) -> List[Violation]:
    violations = []
    split_names = list(splits.keys())
    pairs = (
        [("train", name) for name in split_names if name != "train"]
        if "train" in splits
        else [(split_names[i], split_names[j]) for i in range(len(split_names)) for j in range(i + 1, len(split_names))]
    )

    for name1, name2 in pairs:
        hashes1 = {normalized_prompt_hash(r) for r in splits[name1] if contamination_prompt_text(r).strip()}
        hashes2 = {normalized_prompt_hash(r) for r in splits[name2] if contamination_prompt_text(r).strip()}

        overlap = hashes1.intersection(hashes2)
        if overlap:
            violations.append(Violation(
                "cross_split_prompt_hash_overlap",
                f"Prompt text overlap between {name1} and {name2}",
                list(overlap)
            ))
    return violations

def validate_source_uniqueness(splits: Dict[str, List[dict]]) -> List[Violation]:
    owners: dict[str, tuple[str, str]] = {}
    violations: list[Violation] = []
    for split_name, records in splits.items():
        for index, record in enumerate(records):
            source = source_key(record)
            if source is None:
                continue
            sid = sample_id(record, index)
            owner = owners.get(source)
            if owner is not None:
                prev_split, prev_id = owner
                if split_name != "train" or prev_split != "train":
                    violations.append(Violation(
                        "source_reused_across_eval_boundary",
                        f"Source '{source}' appears in both {prev_split} and {split_name}",
                        [prev_id, sid],
                    ))
            else:
                owners[source] = (split_name, sid)
    return violations

def validate_ngram_contamination(
    splits: Dict[str, List[dict]],
    *,
    code_shingle_size: int,
    prompt_shingle_size: int,
    max_code_containment: float,
    max_prompt_containment: float,
) -> List[Violation]:
    violations: list[Violation] = []
    if "train" not in splits:
        return violations

    def build_index(records: list[dict], extractor, shingle_size: int) -> dict[tuple[str, ...], list[str]]:
        index: dict[tuple[str, ...], list[str]] = {}
        for row_index, row in enumerate(records):
            sid = sample_id(row, row_index)
            for shingle in shingles(extractor(row), shingle_size):
                owners = index.setdefault(shingle, [])
                owners.append(sid)
        return index

    def first_contamination(
        candidate_shingles: set[tuple[str, ...]],
        index: dict[tuple[str, ...], list[str]],
        threshold: float,
    ) -> tuple[str, float] | None:
        if not candidate_shingles:
            return None
        counts: dict[str, int] = {}
        for shingle in candidate_shingles:
            for train_id in index.get(shingle, []):
                counts[train_id] = counts.get(train_id, 0) + 1
        if not counts:
            return None
        train_id, overlap_count = max(counts.items(), key=lambda item: item[1])
        containment = overlap_count / len(candidate_shingles)
        if containment > threshold:
            return train_id, containment
        return None

    train_code_index = build_index(splits["train"], assistant_text, code_shingle_size)
    train_prompt_index = build_index(splits["train"], contamination_prompt_text, prompt_shingle_size)

    for split_name, records in splits.items():
        if split_name == "train":
            continue
        for index, record in enumerate(records):
            sid = sample_id(record, index)
            candidate_code = shingles(assistant_text(record), code_shingle_size)
            code_hit = first_contamination(candidate_code, train_code_index, max_code_containment)
            if code_hit is not None:
                train_id, containment = code_hit
                violations.append(Violation(
                    "code_ngram_contamination",
                    f"{split_name}:{sid} has {containment:.3f} code shingle containment with train:{train_id}",
                    [sid, train_id],
                ))
            candidate_prompt = shingles(contamination_prompt_text(record), prompt_shingle_size)
            prompt_hit = first_contamination(candidate_prompt, train_prompt_index, max_prompt_containment)
            if prompt_hit is not None:
                train_id, containment = prompt_hit
                violations.append(Violation(
                    "prompt_ngram_contamination",
                    f"{split_name}:{sid} has {containment:.3f} prompt shingle containment with train:{train_id}",
                    [sid, train_id],
                ))
    return violations

def write_integrity_manifest(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", help="format: name=path/to.jsonl", required=True)
    parser.add_argument("--out", required=True, help="Path to output manifest JSON")
    parser.add_argument("--code-ngram-size", type=int, default=13)
    parser.add_argument("--prompt-ngram-size", type=int, default=8)
    parser.add_argument("--max-code-containment", type=float, default=0.20)
    parser.add_argument("--max-prompt-containment", type=float, default=0.50)
    parser.add_argument("--allow-source-reuse", action="store_true")
    args = parser.parse_args()
    
    splits = {}
    for s in args.split:
        name, path_str = s.split("=", 1)
        splits[name] = load_jsonl(Path(path_str))
        
    violations = []
    
    for name, records in splits.items():
        violations.extend(validate_no_duplicate_ids(name, records))
        
    violations.extend(validate_no_cross_split_id_overlap(splits))
    violations.extend(validate_no_cross_split_target_hash_overlap(splits))
    violations.extend(validate_no_cross_split_prompt_hash_overlap(splits))
    violations.extend(validate_ngram_contamination(
        splits,
        code_shingle_size=args.code_ngram_size,
        prompt_shingle_size=args.prompt_ngram_size,
        max_code_containment=args.max_code_containment,
        max_prompt_containment=args.max_prompt_containment,
    ))
    if not args.allow_source_reuse:
        violations.extend(validate_source_uniqueness(splits))
    
    payload = {
        "pass": len(violations) == 0,
        "settings": {
            "code_ngram_size": args.code_ngram_size,
            "prompt_ngram_size": args.prompt_ngram_size,
            "max_code_containment": args.max_code_containment,
            "max_prompt_containment": args.max_prompt_containment,
            "allow_source_reuse": args.allow_source_reuse,
        },
        "split_counts": {name: len(records) for name, records in splits.items()},
        "violations": [asdict(v) for v in violations]
    }
    
    write_integrity_manifest(Path(args.out), payload)
    
    if payload["pass"]:
        print("Split integrity validation passed.")
        sys.exit(0)
    else:
        print("Split integrity validation failed:", file=sys.stderr)
        for v in payload["violations"][:50]:
            print(f"  - {v['type']}: {v['message']} ({len(v['ids'])} items)", file=sys.stderr)
        if len(payload["violations"]) > 50:
            print(f"  ... {len(payload['violations']) - 50} more violations in {args.out}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
