#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from tikz_mlx.collapse_probe import run_collapse_probe_suite
from tikz_mlx.model_io import MlxVlmAdapter
from tikz_mlx.prompting import build_generation_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the TikZ collapse probe against an adapter.")
    parser.add_argument("--config", required=True, help="Curriculum stage YAML config.")
    parser.add_argument("--adapter", required=True, help="Adapter checkpoint to probe.")
    parser.add_argument("--out", required=True, help="JSON output path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_id = config.get("model", {}).get("model_id")
    if not model_id:
        print(f"ERROR: model.model_id missing from {config_path}", file=sys.stderr)
        return 1

    model, processor = MlxVlmAdapter.load_model(model_id, args.adapter)
    initial_decoding = config.get("inference", {}).get("initial_decoding", {})
    production_decoding = {
        "temperature": initial_decoding.get("temperature", 0.0),
        "top_p": initial_decoding.get("top_p", 1.0),
        "top_k": initial_decoding.get("top_k", 64),
        "min_p": initial_decoding.get("min_p", 0.05),
        "repetition_penalty": initial_decoding.get("repetition_penalty", 1.2),
    }
    probe_payload = run_collapse_probe_suite(
        model,
        processor,
        build_generation_prompt,
        verbose=True,
        production_decoding=production_decoding,
    )

    payload = {
        **probe_payload,
        "adapter": str(Path(args.adapter)),
        "config": str(config_path),
        "model_id": model_id,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not probe_payload["passed"]:
        print(f"ERROR: collapse probe failed for {args.adapter}; details: {out_path}", file=sys.stderr)
        return 1
    print(f"Collapse probe passed: {args.adapter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
