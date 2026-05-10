import argparse
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

def load_records(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_ids_from_paths(paths: List[Path]) -> set:
    ids = set()
    for p in paths:
        if p.exists():
            for i, r in enumerate(load_records(p)):
                sid = str(r.get("sample_id") or f"row_{i:06d}")
                ids.add(sid)
    return ids

def sample_id(record: dict, index: int = 0) -> str:
    return str(record.get("sample_id") or f"row_{index:06d}")

def group_by_mode(records: List[dict]) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for r in records:
        mode = r.get("metadata", {}).get("generation_mode", "unknown")
        groups[mode].append(r)
    return dict(groups)

def select_mode_balanced(records: List[dict], n: int, seed: int) -> List[dict]:
    groups = group_by_mode(records)
    modes = list(groups.keys())
    modes.sort() # determinism
    
    rng = random.Random(seed)
    for mode in modes:
        rng.shuffle(groups[mode])
        
    selected = []
    mode_idx = 0
    while len(selected) < n and any(groups.values()):
        mode = modes[mode_idx % len(modes)]
        if groups[mode]:
            selected.append(groups[mode].pop(0))
        mode_idx += 1
        
    rng.shuffle(selected)
    return selected

def write_manifest(name: str, records: List[dict], out_dir: Path, source_path: Path, seed: int) -> None:
    jsonl_path = out_dir / f"{name}.jsonl"
    manifest_path = out_dir / f"{name}_manifest.json"
    
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    hasher = hashlib.sha256()
    with jsonl_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
            
    mode_counts = {}
    for r in records:
        mode = r.get("metadata", {}).get("generation_mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
    source_hasher = hashlib.sha256()
    with source_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            source_hasher.update(chunk)
            
    manifest = {
        "name": name,
        "seed": seed,
        "sample_ids": [sample_id(r, i) for i, r in enumerate(records)],
        "source_sha256": source_hasher.hexdigest(),
        "mode_counts": mode_counts,
        "record_count": len(records),
        "manifest_sha256": hasher.hexdigest()
    }
    
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source JSONL to sample from")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude JSONL paths")
    args = parser.parse_args()
    
    source_path = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    records = load_records(source_path)
    
    if args.exclude:
        excluded_ids = load_ids_from_paths([Path(p) for p in args.exclude])
        candidate_records = [r for i, r in enumerate(records) if sample_id(r, i) not in excluded_ids]
    else:
        candidate_records = records
        
    # Generate splits
    # We must chain the sampling so they don't overlap with each other
    rng = random.Random(args.seed)
    
    # 1. sentinel_32
    sentinel = select_mode_balanced(candidate_records, 32, args.seed + 1)
    sentinel_ids = {sample_id(r) for r in sentinel}
    candidate_records = [r for r in candidate_records if sample_id(r) not in sentinel_ids]
    
    # 2. promotion_50
    promotion = select_mode_balanced(candidate_records, 50, args.seed + 2)
    promotion_ids = {sample_id(r) for r in promotion}
    candidate_records = [r for r in candidate_records if sample_id(r) not in promotion_ids]
    
    # 3. rollback_score_50
    rollback = select_mode_balanced(candidate_records, 50, args.seed + 3)
    rollback_ids = {sample_id(r) for r in rollback}
    candidate_records = [r for r in candidate_records if sample_id(r) not in rollback_ids]
    
    # 4. stage4_distribution_probe_50
    probe = select_mode_balanced(candidate_records, 50, args.seed + 4)
    
    write_manifest("sentinel_32", sentinel, out_dir, source_path, args.seed + 1)
    write_manifest("promotion_50", promotion, out_dir, source_path, args.seed + 2)
    write_manifest("rollback_score_50", rollback, out_dir, source_path, args.seed + 3)
    write_manifest("stage4_distribution_probe_50", probe, out_dir, source_path, args.seed + 4)
    
    print(f"Generated 4 eval manifests in {out_dir}")

if __name__ == "__main__":
    main()
