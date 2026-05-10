import argparse
import json
import sys
import glob
from pathlib import Path

def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def checkpoint_is_eligible(metrics: dict) -> bool:
    return (
        metrics.get("deterministic_loop") is False
        and metrics.get("raw_repetition_loop_rate", 1.0) <= 0.02
        and metrics.get("substantive_rate", 0.0) >= 0.80
        and metrics.get("truncation_rate", 1.0) <= 0.10
    )

def checkpoint_sort_key(metrics: dict):
    return (
        metrics.get("raw_repetition_loop_rate", 1.0),          # lower
        -metrics.get("compile_rate", 0.0),                     # higher
        -metrics.get("substantive_rate", 0.0),                 # higher
        metrics.get("avg_code_length_ratio_vs_base", 99.0),    # lower
        -metrics.get("step", 0),                               # later only tie-break
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-evals", required=True, help="Glob pattern or directory for checkpoint evals")
    parser.add_argument("--out", required=True, help="Path to output selected checkpoint JSON")
    args = parser.parse_args()
    
    # Resolve the glob
    if "*" in args.checkpoint_evals or "?" in args.checkpoint_evals:
        paths = glob.glob(args.checkpoint_evals)
    else:
        p = Path(args.checkpoint_evals)
        if p.is_dir():
            paths = list(p.glob("*.json"))
        else:
            paths = [str(p)]
            
    paths = [Path(p) for p in paths if Path(p).is_file()]
    if not paths:
        print(f"Error: No eval files found matching {args.checkpoint_evals}", file=sys.stderr)
        sys.exit(1)
        
    all_candidates = []
    for path in paths:
        metrics = load_metrics(path)
        metrics["_source_file"] = str(path)
        all_candidates.append(metrics)
        
    eligible = [c for c in all_candidates if checkpoint_is_eligible(c)]
    
    if not eligible:
        print("Warning: NO checkpoints are eligible for promotion!", file=sys.stderr)
        out_payload = {
            "selected_checkpoint": None,
            "selected_step": None,
            "reason": "No checkpoints met the strict promotion gates.",
            "metrics": None,
            "all_candidates": all_candidates
        }
    else:
        # Sort eligible
        eligible.sort(key=checkpoint_sort_key)
        best = eligible[0]
        
        # We need to map back to the checkpoint directory from the eval file
        # typically runs/curriculum_stage4/eval_checkpoints/ckpt_050.json -> runs/curriculum_stage4/checkpoint_000050
        # Wait, the user didn't specify exactly how to map it. We will just output the step and the source file.
        
        out_payload = {
            "selected_checkpoint_eval_file": best["_source_file"],
            "selected_step": best.get("step"),
            "reason": "Best candidate passing promotion gates",
            "metrics": best,
            "all_candidates": all_candidates
        }
        print(f"Selected step {best.get('step')} from {best['_source_file']}")
        
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
        
if __name__ == "__main__":
    main()
