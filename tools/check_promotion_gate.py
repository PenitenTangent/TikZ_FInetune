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


def evaluate_promotion_gate(
    data: dict,
    *,
    variant: str = "finetuned",
    base_variant: str = "base",
    max_avg_token_ratio_vs_base: float = 1.6,
    max_avg_code_length_ratio_vs_base: float = 1.6,
) -> dict:
    if variant not in data:
        raise KeyError(f"Variant '{variant}' not found in results.json.")

    m = data[variant]
    base = data.get(base_variant, {})

    violations: list[str] = []

    # ── Zero-tolerance: collapse signals ──────────────────────────────────────
    # Any nonzero rate here is an immediate failure.
    for metric in [
        "preview_environment_rate",
        "assistant_documentclass_rate",
        "decorations_geometric_rate",
        "repetition_loop_rate",
    ]:
        val = m.get(metric, 0.0)
        if val > 0.0:
            violations.append(f"{metric}: {val:.4f} > 0.0  (zero tolerance)")

    # ── Near-zero tolerance: minor collapse signals ───────────────────────────
    for metric, threshold in [
        ("assistant_usepackage_rate", 0.02),
    ]:
        val = m.get(metric, 0.0)
        if val > threshold:
            violations.append(f"{metric}: {val:.4f} > {threshold:.4f} (near-zero tolerance)")

    # ── Bad-pattern pass rate must be >= 98% ──────────────────────────────────
    bpp = m.get("bad_pattern_pass_rate", 0.0)
    if bpp < 0.98:
        violations.append(f"bad_pattern_pass_rate: {bpp:.4f} < 0.98")

    # ── Closing fence: 95%+ must close exactly once ───────────────────────────
    fence_rate = m.get("closing_fence_exactly_once_rate", 0.0)
    if fence_rate < 0.95:
        violations.append(f"closing_fence_exactly_once_rate: {fence_rate:.4f} < 0.95")

    # ── Length gates ──────────────────────────────────────────────────────────
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
    if base_compile is not None:
        # Candidate must be >= 80% of base, with an absolute floor of 70%
        rel_min = max(0.70, 0.80 * base_compile)
        if cand_compile < rel_min:
            violations.append(
                f"compile_rate: {cand_compile:.3f} < {rel_min:.3f} "
                f"(80% of base={base_compile:.3f}, floor=0.70)"
            )
    else:
        # No base available - use absolute floor
        if cand_compile < 0.70:
            violations.append(f"compile_rate: {cand_compile:.3f} < 0.70 (no base; absolute floor)")

    # ── Relative substantive rate ──────────────────────────────────────────────
    base_subst = base.get("substantive_rate", None)
    cand_subst = m.get("substantive_rate", 0.0)
    if base_subst is not None:
        rel_min_s = max(0.85, 0.90 * base_subst)
        if cand_subst < rel_min_s:
            violations.append(
                f"substantive_rate: {cand_subst:.3f} < {rel_min_s:.3f} "
                f"(90% of base={base_subst:.3f}, floor=0.85)"
            )
    else:
        if cand_subst < 0.85:
            violations.append(f"substantive_rate: {cand_subst:.3f} < 0.85 (no base; absolute floor)")

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
            "max_avg_token_ratio_vs_base": max_avg_token_ratio_vs_base,
            "max_avg_code_length_ratio_vs_base": max_avg_code_length_ratio_vs_base,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce promotion gate on TikZ eval results.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing results.json")
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

    gate_result = evaluate_promotion_gate(
        data,
        variant=args.variant,
        base_variant=args.base_variant,
        max_avg_token_ratio_vs_base=args.max_avg_token_ratio_vs_base,
        max_avg_code_length_ratio_vs_base=args.max_avg_code_length_ratio_vs_base,
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
