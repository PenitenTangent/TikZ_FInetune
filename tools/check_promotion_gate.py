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


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce promotion gate on TikZ eval results.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing results.json")
    parser.add_argument("--variant", default="finetuned", help="Variant to check (default: finetuned)")
    parser.add_argument("--base-variant", default="base", help="Base variant key (default: base)")
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

    m = data[args.variant]
    base = data.get(args.base_variant, {})

    violations: list[str] = []

    # ── Zero-tolerance: collapse signals ──────────────────────────────────────
    # Any nonzero rate here is an immediate failure.
    for metric in [
        "preview_environment_rate",
        "assistant_usepackage_rate",
        "assistant_documentclass_rate",
        "decorations_geometric_rate",
        "repetition_loop_rate",
    ]:
        val = m.get(metric, 0.0)
        if val > 0.0:
            violations.append(f"{metric}: {val:.4f} > 0.0  (zero tolerance)")

    # ── Bad-pattern pass rate must be 100% ────────────────────────────────────
    bpp = m.get("bad_pattern_pass_rate", 0.0)
    if bpp < 1.0:
        violations.append(f"bad_pattern_pass_rate: {bpp:.4f} < 1.0")

    # ── Closing fence: 95%+ must close exactly once ───────────────────────────
    fence_rate = m.get("closing_fence_exactly_once_rate", 0.0)
    if fence_rate < 0.95:
        violations.append(f"closing_fence_exactly_once_rate: {fence_rate:.4f} < 0.95")

    # ── Code length ratio vs base ─────────────────────────────────────────────
    length_ratio = m.get("avg_code_length_ratio_vs_base", 999.0)
    if length_ratio > 1.6:
        violations.append(f"avg_code_length_ratio_vs_base: {length_ratio:.3f} > 1.6")

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
        # No base available — use absolute floor
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

    # ── Save results ───────────────────────────────────────────────────────────
    gate_result = {
        "pass": len(violations) == 0,
        "violations": violations,
        "candidate_metrics": m,
        "base_metrics": base,
    }
    gate_path = Path(args.eval_dir) / "promotion_gate_result.json"
    gate_path.write_text(json.dumps(gate_result, indent=2), encoding="utf-8")

    if gate_result["pass"]:
        print("✅ Promotion gate passed.")
        sys.exit(0)
    else:
        print("❌ Promotion gate FAILED:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)


if __name__ == "__main__":
    main()
