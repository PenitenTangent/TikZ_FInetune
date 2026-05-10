import argparse
import json
import sys
from pathlib import Path
from tikz_mlx.bad_patterns import check_bad_patterns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input text file to scan")
    parser.add_argument("--json-out", required=True, help="Output JSON results")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {input_path} does not exist", file=sys.stderr)
        sys.exit(1)
        
    text = input_path.read_text(encoding="utf-8")
    result = check_bad_patterns(text)
    
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    
    if not result["pass"]:
        print("Found bad pattern violations:", file=sys.stderr)
        for v in result["violations"]:
            print(f"  - {v['rule']}: found {v['count']} (max {v['max_allowed']})", file=sys.stderr)
        sys.exit(1)
        
    print("Passed bad patterns check.")
    sys.exit(0)

if __name__ == "__main__":
    main()
