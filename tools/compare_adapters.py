import argparse
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
from safetensors import safe_open
from collections import defaultdict

def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_tensors(path: Path) -> dict:
    tensors = {}
    with safe_open(path, framework="np", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def tensor_stats(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "mean_abs": float(np.abs(arr).mean()),
        "max_abs": float(np.abs(arr).max()),
        "l2": float(np.sqrt((arr * arr).sum())),
    }

def compare_adapter_tensors(prev: dict, curr: dict) -> dict:
    per_tensor = {}
    global_prev_l2_sq = 0.0
    global_curr_l2_sq = 0.0
    global_delta_l2_sq = 0.0
    warnings = []
    
    layer_deltas = defaultdict(float)
    
    for key in curr:
        c_arr = curr[key]
        c_stats = tensor_stats(c_arr)
        global_curr_l2_sq += c_stats["l2"]**2
        
        if key not in prev:
            per_tensor[key] = {"status": "added", "curr": c_stats}
            warnings.append(f"Added tensor {key}")
            continue
            
        p_arr = prev[key]
        p_stats = tensor_stats(p_arr)
        global_prev_l2_sq += p_stats["l2"]**2
        
        delta_arr = c_arr - p_arr
        delta_l2 = float(np.sqrt((delta_arr * delta_arr).sum()))
        global_delta_l2_sq += delta_l2**2
        
        per_tensor[key] = {
            "status": "changed",
            "prev": p_stats,
            "curr": c_stats,
            "delta_l2": delta_l2
        }
        
        import re
        m = re.search(r"layers\.(\d+)", key)
        if m:
            layer_deltas[m.group(1)] += delta_l2
            
    for key in prev:
        if key not in curr:
            per_tensor[key] = {"status": "removed", "prev": tensor_stats(prev[key])}
            warnings.append(f"Removed tensor {key}")
            
    global_prev_l2 = float(np.sqrt(global_prev_l2_sq))
    global_curr_l2 = float(np.sqrt(global_curr_l2_sq))
    global_delta_l2 = float(np.sqrt(global_delta_l2_sq))
    
    layer_deltas_sorted = sorted(layer_deltas.items(), key=lambda x: x[1], reverse=True)
    
    if global_delta_l2 > global_prev_l2 * 2.0:
        warnings.append(f"Massive global delta: {global_delta_l2} vs prev {global_prev_l2}")
        
    return {
        "global": {
            "prev_l2": global_prev_l2,
            "curr_l2": global_curr_l2,
            "delta_l2": global_delta_l2,
            "delta_to_prev_ratio": global_delta_l2 / global_prev_l2 if global_prev_l2 > 0 else 0
        },
        "per_tensor": per_tensor,
        "layer_deltas": dict(layer_deltas_sorted),
        "warnings": warnings
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev", required=True, help="Previous safetensors adapter")
    parser.add_argument("--curr", required=True, help="Current safetensors adapter")
    parser.add_argument("--out", required=True, help="Output JSON")
    args = parser.parse_args()
    
    prev_path = Path(args.prev)
    curr_path = Path(args.curr)
    
    prev_tensors = load_tensors(prev_path)
    curr_tensors = load_tensors(curr_path)
    
    comp = compare_adapter_tensors(prev_tensors, curr_tensors)
    comp["prev_adapter"] = str(prev_path)
    comp["curr_adapter"] = str(curr_path)
    comp["prev_sha256"] = sha256_file(prev_path)
    comp["curr_sha256"] = sha256_file(curr_path)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)

if __name__ == "__main__":
    main()
