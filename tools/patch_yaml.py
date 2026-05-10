import argparse
import sys
from pathlib import Path

def set_nested(config: dict, dotted_key: str, value: any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input YAML file")
    parser.add_argument("--output", required=True, help="Output YAML file")
    parser.add_argument("--set", action="append", required=True, help="format: key.path=value")
    parser.add_argument("--in-place", action="store_true", help="Allow overwriting input")
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if in_path == out_path and not args.in_place:
        print("Error: input and output cannot be the same unless --in-place is set", file=sys.stderr)
        sys.exit(1)
        
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
    except ImportError:
        import yaml
        
    with in_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    for s in args.set:
        key, val_str = s.split("=", 1)
        import yaml as fallback_yaml
        try:
            val = fallback_yaml.safe_load(val_str)
        except Exception:
            val = val_str
        set_nested(config, key, val)
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    print(f"Patched YAML saved to {out_path}")

if __name__ == "__main__":
    main()
