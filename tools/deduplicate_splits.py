import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import Set, Dict, Any

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tikz_mlx.recovery import _single_role_text, _sample_id, iter_jsonl

def normalized_hash(text: str) -> str:
    # Basic normalization: strip whitespace and hash
    normalized = text.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def get_hashes(path: Path) -> Dict[str, Set[str]]:
    ids = set()
    hashes = set()
    if not path.exists():
        return {"ids": ids, "hashes": hashes}
        
    for index, record in enumerate(iter_jsonl(path)):
        sample_id = _sample_id(record, index)
        ids.add(sample_id)
        
        text = _single_role_text(record, "assistant", 1)
        hashes.add(normalized_hash(text))
        
    return {"ids": ids, "hashes": hashes}

def main():
    parser = argparse.ArgumentParser(description="Deduplicate training data against val/gold splits.")
    parser.add_argument("--train", required=True, help="Input training JSONL")
    parser.add_argument("--val", action="append", help="Validation JSONL(s) to check against")
    parser.add_argument("--gold", action="append", help="Gold JSONL(s) to check against")
    parser.add_argument("--output", required=True, help="Cleaned output JSONL")
    args = parser.parse_args()
    
    blacklist_ids = set()
    blacklist_hashes = set()
    
    # Collect all val/gold items
    check_files = (args.val or []) + (args.gold or [])
    for f in check_files:
        path = Path(f)
        if not path.exists():
            print(f"Warning: skip missing check file {f}")
            continue
        print(f"Loading blacklist from {path}...")
        data = get_hashes(path)
        blacklist_ids.update(data["ids"])
        blacklist_hashes.update(data["hashes"])
        
    print(f"Blacklist total: {len(blacklist_ids)} IDs, {len(blacklist_hashes)} unique hashes.")
    
    in_path = Path(args.train)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    kept = 0
    removed_id = 0
    removed_hash = 0
    total = 0
    
    with in_path.open("r", encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
        for index, line in enumerate(f_in):
            line = line.strip()
            if not line: continue
            total += 1
            record = json.loads(line)
            
            sample_id = _sample_id(record, index)
            text = _single_role_text(record, "assistant", 1)
            text_hash = normalized_hash(text)
            
            if sample_id in blacklist_ids:
                removed_id += 1
                continue
            if text_hash in blacklist_hashes:
                removed_hash += 1
                continue
                
            f_out.write(line + "\n")
            kept += 1
            
    print(f"Deduplication complete:")
    print(f"  Total input: {total}")
    print(f"  Removed (ID overlap):   {removed_id}")
    print(f"  Removed (Hash overlap): {removed_hash}")
    print(f"  Total kept:             {kept}")
    print(f"  Cleaned dataset saved to: {out_path}")

if __name__ == "__main__":
    main()
