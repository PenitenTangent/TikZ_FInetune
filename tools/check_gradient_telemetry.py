#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSONL telemetry row: {exc}") from exc
    return rows


def _finite_float(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Telemetry value `{key}` is not numeric: {row[key]!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Telemetry value `{key}` is not finite: {value!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate gradient clipping telemetry from a training run.")
    parser.add_argument("--telemetry", required=True, help="Path to gradient_clip_telemetry.jsonl")
    parser.add_argument("--out", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--window", type=int, default=5, help="Number of recent telemetry rows to inspect.")
    parser.add_argument("--max-clipped-step-rate", type=float, default=1.0)
    parser.add_argument("--min-avg-clip-scale", type=float, default=0.05)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    telemetry_path = Path(args.telemetry)
    if not telemetry_path.exists():
        if args.allow_missing:
            print(f"WARNING: gradient telemetry not found: {telemetry_path}")
            return 0
        print(f"ERROR: gradient telemetry not found: {telemetry_path}", file=sys.stderr)
        return 1

    rows = _load_jsonl(telemetry_path)
    if not rows:
        print(f"ERROR: gradient telemetry is empty: {telemetry_path}", file=sys.stderr)
        return 1

    window = rows[-max(1, args.window):]
    failures: list[str] = []
    clip_rates: list[float] = []
    clip_scales: list[float] = []

    for row in window:
        for key in ("train_loss", "avg_grad_norm", "avg_clip_scale", "clipped_step_rate"):
            if key not in row:
                failures.append(f"missing `{key}` in telemetry row at iteration {row.get('iteration', '?')}")
                continue
            try:
                _finite_float(row, key)
            except RuntimeError as exc:
                failures.append(str(exc))
        if "clipped_step_rate" in row:
            try:
                clip_rates.append(_finite_float(row, "clipped_step_rate"))
            except RuntimeError:
                pass
        if "avg_clip_scale" in row:
            try:
                clip_scales.append(_finite_float(row, "avg_clip_scale"))
            except RuntimeError:
                pass

    mean_clipped_rate = sum(clip_rates) / len(clip_rates) if clip_rates else 1.0
    mean_clip_scale = sum(clip_scales) / len(clip_scales) if clip_scales else 0.0
    if mean_clipped_rate > args.max_clipped_step_rate:
        failures.append(
            f"recent clipped_step_rate mean {mean_clipped_rate:.3f} exceeds {args.max_clipped_step_rate:.3f}"
        )
    if mean_clip_scale < args.min_avg_clip_scale:
        failures.append(f"recent avg_clip_scale mean {mean_clip_scale:.3f} is below {args.min_avg_clip_scale:.3f}")

    summary = {
        "passed": not failures,
        "telemetry_path": str(telemetry_path),
        "rows": len(rows),
        "window": len(window),
        "mean_clipped_step_rate": mean_clipped_rate,
        "mean_avg_clip_scale": mean_clip_scale,
        "failures": failures,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if failures:
        print("ERROR: gradient telemetry gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        "Gradient telemetry gate passed "
        f"(mean clipped={mean_clipped_rate:.1%}, mean clip scale={mean_clip_scale:.3f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
