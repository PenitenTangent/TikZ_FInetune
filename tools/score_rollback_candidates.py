import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

def load_promotion_gate(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def score_candidates(eval_results_paths: List[Path], gate_config: Dict[str, Any]) -> Dict[str, Any]:
    candidates = []
    
    for path in eval_results_paths:
        if not path.exists():
            print(f"Warning: Result path {path} does not exist", file=sys.stderr)
            continue
            
        with path.open("r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {path}", file=sys.stderr)
                continue
                
        base_stats = results.get("base")
        if not base_stats:
            print(f"Warning: No 'base' stats found in {path}", file=sys.stderr)
            continue
            
        base_length = base_stats.get("avg_code_length", 1.0)
        if base_length == 0:
            base_length = 1.0
            
        # Consider all checkpoints except "base" as candidates
        for label, stats in results.items():
            if label == "base" or not isinstance(stats, dict) or "compile_rate" not in stats:
                continue
                
            avg_len_ratio = stats.get("avg_code_length", 0) / base_length
            
            # Check gates (assuming these are normal eval sets for now)
            gates = gate_config.get("promotion_gate", {}).get("normal", {})
            global_gates = gate_config.get("promotion_gate", {}).get("global", {})
            
            passed_compile = stats.get("compile_rate", 0) >= gates.get("min_compile_rate", 0)
            passed_loop = stats.get("repetition_loop_rate", 1.0) <= gates.get("max_raw_repetition_loop_rate", 1.0)
            passed_subst = stats.get("substantive_rate", 0) >= gates.get("min_substantive_rate", 0)
            passed_trunc = stats.get("truncation_rate", 1.0) <= global_gates.get("max_truncation_rate", 1.0)
            passed_ratio = avg_len_ratio <= global_gates.get("max_avg_length_ratio_vs_base", 999.0)
            
            all_passed = passed_compile and passed_loop and passed_subst and passed_trunc and passed_ratio
            
            candidates.append({
                "label": label,
                "source_file": str(path),
                "stats": stats,
                "avg_length_ratio": avg_len_ratio,
                "passed_gates": all_passed,
                "gate_details": {
                    "compile": passed_compile,
                    "loop": passed_loop,
                    "substantive": passed_subst,
                    "truncation": passed_trunc,
                    "length_ratio": passed_ratio
                }
            })
            
    # Filter candidates that passed all gates
    valid_candidates = [c for c in candidates if c["passed_gates"]]
    
    if not valid_candidates:
        return {"status": "failure", "reason": "No candidates passed the promotion gates.", "all_candidates": candidates}
        
    # Rank candidates: 
    # 1. Compile rate (descending)
    # 2. Lower repetition loop rate (ascending)
    # 3. Substantive rate (descending)
    # 4. Avg length ratio vs base (closer to 1.0 is better, but we just want it to not be too long)
    valid_candidates.sort(key=lambda c: (
        c["stats"].get("compile_rate", 0),
        -c["stats"].get("repetition_loop_rate", 0),
        c["stats"].get("substantive_rate", 0),
        -c["avg_length_ratio"]
    ), reverse=True)
    
    winner = valid_candidates[0]
    
    return {
        "status": "success",
        "selected_adapter_label": winner["label"],
        "source_file": winner["source_file"],
        "stats": winner["stats"],
        "rationale": "Ranked highest in compile rate, followed by low loop rate and high substantive rate.",
        "all_valid_candidates": valid_candidates
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", nargs="+", required=True, help="Path to one or more results.json from ab_eval.py")
    parser.add_argument("--gate-config", required=True, help="Path to promotion_gate.yaml")
    parser.add_argument("--out-json", required=True, help="Output JSON path for the selection rationale")
    args = parser.parse_args()
    
    gate_config = load_promotion_gate(Path(args.gate_config))
    eval_paths = [Path(p) for p in args.eval_results]
    
    result = score_candidates(eval_paths, gate_config)
    
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    if result["status"] == "success":
        print(f"Selected candidate: {result['selected_adapter_label']} from {result['source_file']}")
        print(f"Stats: {json.dumps(result['stats'], indent=2)}")
        sys.exit(0)
    else:
        print(f"Failed to find a rollback candidate: {result['reason']}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
