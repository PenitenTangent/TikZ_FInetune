import argparse
import json
import hashlib
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Dict, Any, Optional
import numpy as np

MARKERS = [
    r"\\PreviewEnvironment",
    r"\\usepackage",
    r"\\usetikzlibrary",
    r"\\documentclass",
    r"\\begin\{document\}",
    r"\\end\{document\}",
    r"decorations\.geometric",
    r"shapes\.geometric",
    r"0\.geometric",
]

@dataclass
class DatasetStats:
    path: str
    sha256: str
    record_count: int
    mode_counts: Dict[str, int]
    assistant_length_quantiles: Dict[str, int]
    token_marker_counts: Dict[str, int]
    tikz_command_counts: Dict[str, int]
    rejection_counts: Optional[Dict[str, int]] = None
    compile_pass_rate: Optional[float] = None

def get_assistant_text(record: dict) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                return "".join(c.get("text", "") for c in content if c.get("type") == "text")
            return content
    return ""

def compute_dataset_stats(path: Path) -> DatasetStats:
    sha256 = hashlib.sha256()
    record_count = 0
    mode_counts = Counter()
    lengths = []
    marker_counts = Counter()
    cmd_counts = Counter()
    
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
            
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            record_count += 1
            mode = record.get("metadata", {}).get("generation_mode", "unknown")
            mode_counts[mode] += 1
            
            text = get_assistant_text(record)
            lengths.append(len(text))
            
            for marker in MARKERS:
                marker_counts[marker] += len(re.findall(marker, text))
                
            cmds = re.findall(r"\\[A-Za-z@]+", text)
            for cmd in cmds:
                cmd_counts[cmd] += 1
                
    quantiles = {}
    if lengths:
        q_vals = np.quantile(lengths, [0.5, 0.9, 0.95, 0.99])
        quantiles = {
            "p50": int(q_vals[0]),
            "p90": int(q_vals[1]),
            "p95": int(q_vals[2]),
            "p99": int(q_vals[3]),
        }
        
    return DatasetStats(
        path=str(path),
        sha256=sha256.hexdigest(),
        record_count=record_count,
        mode_counts=dict(mode_counts),
        assistant_length_quantiles=quantiles,
        token_marker_counts=dict(marker_counts),
        tikz_command_counts=dict(cmd_counts.most_common(50)), # keep top 50
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input dataset jsonl")
    parser.add_argument("--out", required=True, help="Output JSON metrics")
    args = parser.parse_args()
    
    stats = compute_dataset_stats(Path(args.input))
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

if __name__ == "__main__":
    main()
