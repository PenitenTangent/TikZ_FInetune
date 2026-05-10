import argparse
import json
import random
from pathlib import Path

def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def build_bridge_split(stage3_path: Path, stage4_path: Path, out_path: Path, seed: int = 42):
    print(f"Loading Stage 3 from {stage3_path}...")
    stage3 = load_records(stage3_path)
    
    print(f"Loading Stage 4 from {stage4_path}...")
    stage4 = load_records(stage4_path)
    
    # Sort by token length
    def get_length(rec):
        return rec.get("metadata", {}).get("token_length", 512)
        
    stage3.sort(key=get_length)
    stage4.sort(key=get_length)
    
    # Take top 50% longest of stage 3
    late_stage3 = stage3[len(stage3)//2:]
    
    # Take bottom 50% shortest of stage 4
    early_stage4 = stage4[:len(stage4)//2]
    
    bridge = late_stage3 + early_stage4
    
    # Shuffle the bridge
    rng = random.Random(seed)
    rng.shuffle(bridge)
    
    print(f"Bridge split contains {len(bridge)} examples ({len(late_stage3)} from Stage 3, {len(early_stage4)} from Stage 4).")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in bridge:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", required=True, help="Path to clean stage 3 JSONL")
    parser.add_argument("--stage4", required=True, help="Path to clean stage 4 JSONL")
    parser.add_argument("--out", required=True, help="Path to output bridge JSONL")
    args = parser.parse_args()
    
    build_bridge_split(Path(args.stage3), Path(args.stage4), Path(args.out))

if __name__ == "__main__":
    main()
