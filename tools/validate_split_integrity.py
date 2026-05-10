import argparse
import json
import hashlib
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

def assistant_text(record: dict) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                return "".join(c.get("text", "") for c in content if c.get("type") == "text")
            return content
    return ""

def normalized_target_hash(record: dict) -> str:
    text = assistant_text(record).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
    
    for i in range(len(split_names)):
        name1 = split_names[i]
        ids1 = {sample_id(r, j) for j, r in enumerate(splits[name1])}
        for j in range(i + 1, len(split_names)):
            name2 = split_names[j]
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
    
    for i in range(len(split_names)):
        name1 = split_names[i]
        hashes1 = {normalized_target_hash(r) for r in splits[name1]}
        for j in range(i + 1, len(split_names)):
            name2 = split_names[j]
            hashes2 = {normalized_target_hash(r) for r in splits[name2]}
            
            overlap = hashes1.intersection(hashes2)
            if overlap:
                violations.append(Violation(
                    "cross_split_target_hash_overlap",
                    f"Target text overlap between {name1} and {name2}",
                    list(overlap)
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
    
    payload = {
        "pass": len(violations) == 0,
        "violations": [asdict(v) for v in violations]
    }
    
    write_integrity_manifest(Path(args.out), payload)
    
    if payload["pass"]:
        print("Split integrity validation passed.")
        sys.exit(0)
    else:
        print("Split integrity validation failed:", file=sys.stderr)
        for v in payload["violations"]:
            print(f"  - {v['type']}: {v['message']} ({len(v['ids'])} items)", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
