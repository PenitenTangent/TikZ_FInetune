#!/usr/bin/env python3
"""Enforce relative + absolute promotion thresholds on A/B eval results.

Requires both 'base' and 'finetuned' (or --variant) to be present in results.json.
Uses relative thresholds anchored to base model performance, preventing promotion
of a model that is worse than base even if it clears absolute minimums.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


DEFAULT_GATE_THRESHOLDS = {
    "max_preview_environment_rate": 0.0,
    "max_assistant_documentclass_rate": 0.0,
    "max_decorations_geometric_rate": 0.0,
    "max_repetition_loop_rate": 0.0,
    "max_assistant_usepackage_rate": 0.02,
    "min_bad_pattern_pass_rate": 0.98,
    "min_closing_fence_exactly_once_rate": 0.95,
    "min_compile_rate": 0.70,
    "min_substantive_rate": 0.85,
    "relative_compile_rate_floor": 0.80,
    "relative_substantive_rate_floor": 0.90,
    "max_avg_token_ratio_vs_base": 1.6,
    "max_avg_code_length_ratio_vs_base": 1.6,
}


def load_gate_thresholds(path: Path, stage: str) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    gates = payload.get("promotion_gate", {})
    if stage not in gates and stage != "normal":
        raise KeyError(f"Promotion gate stage '{stage}' not found in {path}.")

    thresholds = dict(DEFAULT_GATE_THRESHOLDS)
    thresholds.update(gates.get("global", {}))
    thresholds.update(gates.get("normal", {}))
    thresholds.update(gates.get(stage, {}))

    # Backward-compatible aliases for older config revisions.
    if "min_fence_ok_rate" in thresholds:
        thresholds["min_closing_fence_exactly_once_rate"] = thresholds["min_fence_ok_rate"]
    if "max_raw_repetition_loop_rate" in thresholds:
        thresholds["max_repetition_loop_rate"] = thresholds["max_raw_repetition_loop_rate"]
    if "max_avg_length_ratio_vs_base" in thresholds:
        stage_gate = gates.get(stage, {})
        normal_gate = gates.get("normal", {})
        if (
            "max_avg_token_ratio_vs_base" not in stage_gate
            and "max_avg_token_ratio_vs_base" not in normal_gate
        ):
            thresholds["max_avg_token_ratio_vs_base"] = thresholds["max_avg_length_ratio_vs_base"]
        if (
            "max_avg_code_length_ratio_vs_base" not in stage_gate
            and "max_avg_code_length_ratio_vs_base" not in normal_gate
        ):
            thresholds["max_avg_code_length_ratio_vs_base"] = thresholds["max_avg_length_ratio_vs_base"]
    return thresholds


def evaluate_promotion_gate(
    data: dict,
    *,
    variant: str = "finetuned",
    base_variant: str = "base",
    max_avg_token_ratio_vs_base: float = 1.6,
    max_avg_code_length_ratio_vs_base: float = 1.6,
    thresholds: dict | None = None,
) -> dict:
    base_thresholds = {
        **DEFAULT_GATE_THRESHOLDS,
        "max_avg_token_ratio_vs_base": max_avg_token_ratio_vs_base,
        "max_avg_code_length_ratio_vs_base": max_avg_code_length_ratio_vs_base,
    }
    thresholds = {**base_thresholds, **(thresholds or {})}
    max_avg_token_ratio_vs_base = float(
        thresholds.get("max_avg_token_ratio_vs_base", max_avg_token_ratio_vs_base)
    )
    max_avg_code_length_ratio_vs_base = float(
        thresholds.get("max_avg_code_length_ratio_vs_base", max_avg_code_length_ratio_vs_base)
    )

    if variant not in data:
        raise KeyError(f"Variant '{variant}' not found in results.json.")

    m = data[variant]
    base = data.get(base_variant, {})

    violations: list[str] = []

    # ── Collapse signals ──────────────────────────────────────────────────────
    for metric, threshold_key in [
        ("preview_environment_rate", "max_preview_environment_rate"),
        ("assistant_documentclass_rate", "max_assistant_documentclass_rate"),
        ("decorations_geometric_rate", "max_decorations_geometric_rate"),
        ("repetition_loop_rate", "max_repetition_loop_rate"),
    ]:
        threshold = float(thresholds.get(threshold_key, 0.0))
        val = m.get(metric, 0.0)
        if val > threshold:
            violations.append(f"{metric}: {val:.4f} > {threshold:.4f}")

    # ── Near-zero tolerance: minor collapse signals ───────────────────────────
    for metric, threshold in [
        ("assistant_usepackage_rate", float(thresholds.get("max_assistant_usepackage_rate", 0.02))),
    ]:
        val = m.get(metric, 0.0)
        if val > threshold:
            violations.append(f"{metric}: {val:.4f} > {threshold:.4f} (near-zero tolerance)")

    # ── Bad-pattern pass rate must be >= 98% ──────────────────────────────────
    bpp = m.get("bad_pattern_pass_rate", 0.0)
    min_bad_pattern_pass_rate = float(thresholds.get("min_bad_pattern_pass_rate", 0.98))
    if bpp < min_bad_pattern_pass_rate:
        violations.append(f"bad_pattern_pass_rate: {bpp:.4f} < {min_bad_pattern_pass_rate:.4f}")

    # ── Closing fence ─────────────────────────────────────────────────────────
    fence_rate = m.get("closing_fence_exactly_once_rate", 0.0)
    min_fence_rate = float(thresholds.get("min_closing_fence_exactly_once_rate", 0.95))
    if fence_rate < min_fence_rate:
        violations.append(
            f"closing_fence_exactly_once_rate: {fence_rate:.4f} < {min_fence_rate:.4f}"
        )

    # ── Length gates ──────────────────────────────────────────────────────────
    max_truncation_rate = thresholds.get("max_truncation_rate")
    if max_truncation_rate is not None:
        truncation_rate = m.get("truncation_rate", 0.0)
        max_truncation_rate = float(max_truncation_rate)
        if truncation_rate > max_truncation_rate:
            violations.append(f"truncation_rate: {truncation_rate:.4f} > {max_truncation_rate:.4f}")

    length_ratio = m.get("avg_code_length_ratio_vs_base", 999.0)
    if length_ratio > max_avg_code_length_ratio_vs_base:
        violations.append(
            f"avg_code_length_ratio_vs_base: {length_ratio:.3f} > "
            f"{max_avg_code_length_ratio_vs_base:.3f}"
        )
    for metric in ["avg_raw_token_ratio_vs_base", "avg_code_token_ratio_vs_base"]:
        ratio = m.get(metric, None)
        if ratio is not None and ratio > max_avg_token_ratio_vs_base:
            violations.append(f"{metric}: {ratio:.3f} > {max_avg_token_ratio_vs_base:.3f}")

    # ── Relative compile rate ──────────────────────────────────────────────────
    base_compile = base.get("compile_rate", None)
    cand_compile = m.get("compile_rate", 0.0)
    min_compile_rate = float(thresholds.get("min_compile_rate", 0.70))
    relative_compile_rate_floor = float(thresholds.get("relative_compile_rate_floor", 0.80))
    if base_compile is not None:
        rel_min = max(min_compile_rate, relative_compile_rate_floor * base_compile)
        if cand_compile < rel_min:
            violations.append(
                f"compile_rate: {cand_compile:.3f} < {rel_min:.3f} "
                f"({relative_compile_rate_floor:.0%} of base={base_compile:.3f}, "
                f"floor={min_compile_rate:.3f})"
            )
    else:
        if cand_compile < min_compile_rate:
            violations.append(
                f"compile_rate: {cand_compile:.3f} < {min_compile_rate:.3f} "
                "(no base; absolute floor)"
            )

    # ── Relative substantive rate ──────────────────────────────────────────────
    base_subst = base.get("substantive_rate", None)
    cand_subst = m.get("substantive_rate", 0.0)
    min_substantive_rate = float(thresholds.get("min_substantive_rate", 0.85))
    relative_substantive_rate_floor = float(thresholds.get("relative_substantive_rate_floor", 0.90))
    if base_subst is not None:
        rel_min_s = max(min_substantive_rate, relative_substantive_rate_floor * base_subst)
        if cand_subst < rel_min_s:
            violations.append(
                f"substantive_rate: {cand_subst:.3f} < {rel_min_s:.3f} "
                f"({relative_substantive_rate_floor:.0%} of base={base_subst:.3f}, "
                f"floor={min_substantive_rate:.3f})"
            )
    else:
        if cand_subst < min_substantive_rate:
            violations.append(
                f"substantive_rate: {cand_subst:.3f} < {min_substantive_rate:.3f} "
                "(no base; absolute floor)"
            )

    # ── Candidate must not be worse than base on repetition / bad patterns ─────
    if base:
        base_rep = base.get("repetition_loop_rate", 0.0)
        if m.get("repetition_loop_rate", 0.0) > base_rep:
            violations.append(
                f"repetition_loop_rate: {m.get('repetition_loop_rate', 0):.4f} "
                f"> base={base_rep:.4f}"
            )

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "candidate_metrics": m,
        "base_metrics": base,
        "thresholds": {
            **thresholds,
            "max_avg_token_ratio_vs_base": max_avg_token_ratio_vs_base,
            "max_avg_code_length_ratio_vs_base": max_avg_code_length_ratio_vs_base,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce promotion gate on TikZ eval results.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing results.json")
    parser.add_argument("--gate-config", default="configs/promotion_gate.yaml")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "normal"],
    )
    parser.add_argument("--variant", default="finetuned", help="Variant to check (default: finetuned)")
    parser.add_argument("--base-variant", default="base", help="Base variant key (default: base)")
    parser.add_argument("--max-avg-token-ratio-vs-base", type=float, default=1.6)
    parser.add_argument("--max-avg-code-length-ratio-vs-base", type=float, default=1.6)
    args = parser.parse_args()

    results_path = Path(args.eval_dir) / "results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.", file=sys.stderr)
        sys.exit(1)

    with results_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if args.variant not in data:
        print(f"ERROR: Variant '{args.variant}' not found in results.json.", file=sys.stderr)
        sys.exit(1)

    try:
        thresholds = load_gate_thresholds(Path(args.gate_config), args.stage)
    except Exception as exc:
        print(f"ERROR: failed to load promotion gate config: {exc}", file=sys.stderr)
        sys.exit(1)

    gate_result = evaluate_promotion_gate(
        data,
        variant=args.variant,
        base_variant=args.base_variant,
        max_avg_token_ratio_vs_base=args.max_avg_token_ratio_vs_base,
        max_avg_code_length_ratio_vs_base=args.max_avg_code_length_ratio_vs_base,
        thresholds=thresholds,
    )
    gate_path = Path(args.eval_dir) / "promotion_gate_result.json"
    gate_path.write_text(json.dumps(gate_result, indent=2), encoding="utf-8")

    if gate_result["pass"]:
        print("✅ Promotion gate passed.")
        sys.exit(0)
    else:
        print("❌ Promotion gate FAILED:")
        for v in gate_result["violations"]:
            print(f"  - {v}")
        sys.exit(1)


if __name__ == "__main__":
    main()
