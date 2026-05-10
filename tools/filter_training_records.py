import argparse
import json
import sys
from pathlib import Path
from tikz_mlx.bad_patterns import check_bad_patterns
from tikz_mlx.token_stats import boilerplate_score
from tikz_mlx.recovery import substantive_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL dataset")
    parser.add_argument("--output", required=True, help="Filtered output JSONL")
    parser.add_argument("--rejected", required=False, help="Path to save rejected records JSONL")
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    rej_path = Path(args.rejected) if args.rejected else None
    
    if not in_path.exists():
        print(f"Error: {in_path} does not exist.", file=sys.stderr)
        sys.exit(1)
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rej_path:
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        
    accepted_count = 0
    rejected_count = 0
    
    with in_path.open("r", encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:
         
        f_rej = rej_path.open("w", encoding="utf-8") if rej_path else None
        
        try:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                
                # Get assistant text
                text = ""
                for msg in record.get("messages", []):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
                        else:
                            text = content
                        break
                        
                # Evaluate
                bad_pats = check_bad_patterns(text)
                substantive = substantive_features(text)
                b_score = boilerplate_score(text)
                
                # Decision logic
                reject_reasons = []
                if not bad_pats["pass"]:
                    reject_reasons.append("bad_patterns")
                if not substantive["substantive_pass"]:
                    reject_reasons.append("insufficient_substantive_score")
                if b_score > 5.0: # Arbitrary high boilerplate threshold
                    reject_reasons.append("high_boilerplate")
                    
                if reject_reasons:
                    rejected_count += 1
                    if f_rej:
                        record["_reject_reasons"] = reject_reasons
                        f_rej.write(json.dumps(record) + "\n")
                else:
                    accepted_count += 1
                    f_out.write(json.dumps(record) + "\n")
        finally:
            if f_rej:
                f_rej.close()
                
    print(f"Filtering complete. Accepted: {accepted_count}, Rejected: {rejected_count}")

if __name__ == "__main__":
    main()
