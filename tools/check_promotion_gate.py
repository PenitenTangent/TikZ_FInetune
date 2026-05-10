#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Enforce hard quality gates on TikZ eval results.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing results.json")
    parser.add_argument("--variant", default="finetuned", help="Variant to check (default: finetuned)")
    args = parser.parse_args()

    results_path = Path(args.eval_dir) / "results.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        sys.exit(1)

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if args.variant not in data:
        print(f"Error: Variant '{args.variant}' not found in results.json.")
        sys.exit(1)

    metrics = data[args.variant]
    
    # Thresholds
    thresholds = {
        "bad_pattern_pass_rate": 1.0,
        "preview_environment_rate": 0.0,
        "assistant_usepackage_rate": 0.0,
        "raw_repetition_loop_rate": 0.02, # Allow tiny fraction of loops if others are perfect
        "compile_rate_min": 0.3,
        "substantive_rate_min": 0.75,
        "avg_code_length_ratio_max": 2.5
    }

    violations = []

    if metrics.get("bad_pattern_pass_rate", 0) < thresholds["bad_pattern_pass_rate"]:
        violations.append(f"bad_pattern_pass_rate: {metrics['bad_pattern_pass_rate']} < {thresholds['bad_pattern_pass_rate']}")

    if metrics.get("preview_environment_rate", 1) > thresholds["preview_environment_rate"]:
        violations.append(f"preview_environment_rate: {metrics['preview_environment_rate']} > {thresholds['preview_environment_rate']}")

    if metrics.get("assistant_usepackage_rate", 1) > thresholds["assistant_usepackage_rate"]:
        violations.append(f"assistant_usepackage_rate: {metrics['assistant_usepackage_rate']} > {thresholds['assistant_usepackage_rate']}")

    if metrics.get("repetition_loop_rate", 1) > thresholds["raw_repetition_loop_rate"]:
        violations.append(f"repetition_loop_rate: {metrics['repetition_loop_rate']} > {thresholds['raw_repetition_loop_rate']}")

    if metrics.get("compile_rate", 0) < thresholds["compile_rate_min"]:
        violations.append(f"compile_rate: {metrics['compile_rate']} < {thresholds['compile_rate_min']}")

    if metrics.get("substantive_rate", 0) < thresholds["substantive_rate_min"]:
        violations.append(f"substantive_rate: {metrics['substantive_rate']} < {thresholds['substantive_rate_min']}")

    if metrics.get("avg_code_length_ratio_vs_base", 10) > thresholds["avg_code_length_ratio_max"]:
        violations.append(f"avg_code_length_ratio_vs_base: {metrics['avg_code_length_ratio_vs_base']} > {thresholds['avg_code_length_ratio_max']}")

    result = {
        "pass": len(violations) == 0,
        "violations": violations,
        "metrics": metrics
    }

    gate_path = Path(args.eval_dir) / "promotion_gate_result.json"
    gate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result["pass"]:
        print("✅ Promotion gate passed.")
        sys.exit(0)
    else:
        print("❌ Promotion gate FAILED:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)

if __name__ == "__main__":
    main()
