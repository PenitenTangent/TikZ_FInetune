import json
import subprocess
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

def run_eval(config: str, adapter: str, num_samples: int, out_dir: Path):
    cmd = [
        "python3", "tools/ab_eval.py",
        "--config", config,
        "--adapter-path", adapter,
        "--num-samples", str(num_samples),
        "--seed", "42",
        "--max-tokens", "2048",
        "--out-dir", str(out_dir)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    results_path = out_dir / "results.json"
    with open(results_path, "r") as f:
        return json.load(f).get("finetuned", {})

def is_collapsed(metrics: dict) -> list[str]:
    reasons = []
    if metrics.get("preview_environment_rate", 0) > 0:
        reasons.append("preview_environment_rate > 0")
    if metrics.get("assistant_usepackage_rate", 0) > 0:
        reasons.append("assistant_usepackage_rate > 0")
    if metrics.get("repetition_loop_rate", 0) > 0.05:
        reasons.append("repetition_loop_rate > 0.05")
    if metrics.get("avg_code_length_ratio_vs_base", 1.0) > 2.0:
        reasons.append("avg_code_length_ratio_vs_base > 2.0")
    return reasons

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    checkpoints = sorted(list(run_dir.glob("*_adapters.safetensors")), key=lambda x: x.name)
    
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return
        
    print(f"Found {len(checkpoints)} checkpoints. Starting localization...")
    
    last_good = None
    first_bad = None
    failure_reasons = []
    
    for ckpt in tqdm(checkpoints):
        step_dir = run_dir / "eval_localization" / ckpt.stem
        step_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            metrics = run_eval(args.config, str(ckpt), args.num_samples, step_dir)
            reasons = is_collapsed(metrics)
            if reasons:
                first_bad = ckpt
                failure_reasons = reasons
                break
            else:
                last_good = ckpt
        except Exception as e:
            print(f"Failed to evaluate {ckpt}: {e}")
            continue
            
    if first_bad:
        print("\n" + "="*40)
        print("COLLAPSE LOCALIZED")
        print("="*40)
        print(f"Last good checkpoint:  {last_good.name if last_good else 'None'}")
        print(f"First bad checkpoint:   {first_bad.name}")
        print(f"Failure reasons:        {', '.join(failure_reasons)}")
        print("="*40)
    else:
        print("\nNo collapse detected in any checkpoint.")

if __name__ == "__main__":
    main()
