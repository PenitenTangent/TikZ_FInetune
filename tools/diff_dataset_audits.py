import argparse
import json
from pathlib import Path
from typing import Dict, Any

def load_audit(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def compare_stats(raw: dict, clean: dict, probe: dict) -> dict:
    warnings = []
    
    raw_count = raw.get("record_count", 0)
    clean_count = clean.get("record_count", 0)
    
    if raw_count > 0:
        retention = clean_count / raw_count
        if retention < 0.20:
            warnings.append(f"clean dataset loses >80% of records (retention {retention:.1%})")
            
    mode_counts = clean.get("mode_counts", {})
    for mode, count in mode_counts.items():
        if clean_count > 0 and count / clean_count > 0.70:
            warnings.append(f"one mode exceeds 70%: {mode} ({count / clean_count:.1%})")
            
    raw_p99 = raw.get("assistant_length_quantiles", {}).get("p99", 0)
    clean_p99 = clean.get("assistant_length_quantiles", {}).get("p99", 0)
    
    if raw_p99 > 0 and clean_p99 > raw_p99 * 1.5:
        warnings.append(f"assistant length p99 explodes: raw {raw_p99} -> clean {clean_p99}")
        
    markers = clean.get("token_marker_counts", {})
    bad_marker_keys = [
        "\\\\PreviewEnvironment", "0\\.geometric", "decorations\\.geometric"
    ]
    for key in bad_marker_keys:
        if markers.get(key, 0) > 0:
            warnings.append(f"bad markers remain in clean set: {key} ({markers[key]} instances)")
            
    # Probe distribution comparison (modes)
    probe_modes = probe.get("mode_counts", {})
    probe_count = probe.get("record_count", 0)
    if probe_count > 0 and clean_count > 0:
        for mode in probe_modes:
            probe_ratio = probe_modes[mode] / probe_count
            clean_ratio = mode_counts.get(mode, 0) / clean_count
            if abs(probe_ratio - clean_ratio) > 0.30:
                warnings.append(f"probe distribution differs strongly from clean Stage4 for mode {mode}: "
                                f"clean {clean_ratio:.1%} vs probe {probe_ratio:.1%}")

    return {
        "raw": raw,
        "clean": clean,
        "probe": probe,
        "diff_raw_to_clean": {
            "record_loss": raw_count - clean_count,
            "retention_rate": clean_count / raw_count if raw_count > 0 else 0
        },
        "warnings": warnings
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Path to raw dataset stats JSON")
    parser.add_argument("--clean", required=True, help="Path to clean dataset stats JSON")
    parser.add_argument("--probe", required=False, help="Path to probe dataset stats JSON")
    parser.add_argument("--out", required=True, help="Path to output diff JSON")
    args = parser.parse_args()
    
    raw_stats = load_audit(Path(args.raw))
    clean_stats = load_audit(Path(args.clean))
    probe_stats = load_audit(Path(args.probe)) if args.probe else {}
    
    diff = compare_stats(raw_stats, clean_stats, probe_stats)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2)

if __name__ == "__main__":
    main()
