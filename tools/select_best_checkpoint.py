import argparse
import hashlib
import json
import re
import sys
import glob
from pathlib import Path

def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def _resolve_checkpoint_path(metrics: dict, eval_file: Path) -> Path | None:
    """Try to find the actual adapter weights path for an eval record.

    Priority order:
    1. ``checkpoint_path`` field embedded in the eval JSON.
    2. Inferred from the eval filename step number: looks for
       ``{step:06d}_adapters.safetensors`` in the parent of the eval dir.
    """
    if "checkpoint_path" in metrics:
        p = Path(metrics["checkpoint_path"])
        if p.exists():
            return p

    # Attempt filename-based inference: eval files live in e.g.
    # runs/curriculum_stage4/eval_checkpoints/ckpt_000500.json
    # The adapter is at runs/curriculum_stage4/000500_adapters.safetensors
    m = re.search(r"(\d{4,})", eval_file.stem)
    if m:
        step = int(m.group(1))
        # Walk up to find candidate checkpoint dirs (parent of eval_checkpoints/)
        for candidate_dir in [eval_file.parent, eval_file.parent.parent]:
            candidate = candidate_dir / f"{step:06d}_adapters.safetensors"
            if candidate.exists():
                return candidate
    return None

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
            "selected_checkpoint_path": None,
            "selected_adapter_sha256": None,
            "eligible": False,
            "selected_step": None,
            "reason": "No checkpoints met the strict promotion gates.",
            "metrics": None,
            "all_candidates": all_candidates
        }
    else:
        # Sort eligible
        eligible.sort(key=checkpoint_sort_key)
        best = eligible[0]
        eval_file = Path(best["_source_file"])

        checkpoint_path = _resolve_checkpoint_path(best, eval_file)
        adapter_sha256 = None
        if checkpoint_path is not None:
            try:
                adapter_sha256 = _sha256(checkpoint_path)
            except OSError:
                checkpoint_path = None

        if checkpoint_path is None:
            print(
                f"Warning: Could not resolve checkpoint file for step {best.get('step')} "
                f"(eval: {eval_file}). Embed 'checkpoint_path' in the eval JSON to fix this.",
                file=sys.stderr,
            )

        out_payload = {
            "selected_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "selected_adapter_sha256": adapter_sha256,
            "eligible": True,
            "selected_step": best.get("step"),
            "reason": "Best candidate passing promotion gates",
            "metrics": best,
            "all_candidates": all_candidates
        }
        print(f"Selected step {best.get('step')} → {checkpoint_path or '(path unknown)'}")
        
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    # Non-zero exit if no eligible checkpoint was found
    if not out_payload["eligible"]:
        sys.exit(1)
        
if __name__ == "__main__":
    main()
