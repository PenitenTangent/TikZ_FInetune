#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


_CHECKPOINT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _checkpoint_sort_key(path: Path) -> int:
    match = _CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoints(checkpoint_dir: Path, max_candidates: int) -> list[Path]:
    candidates = [
        path for path in checkpoint_dir.glob("*_adapters.safetensors")
        if _CHECKPOINT_RE.match(path.name)
    ]
    candidates.sort(key=_checkpoint_sort_key, reverse=True)
    return candidates[:max(0, max_candidates)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run stage gates from newest to oldest and select the latest adapter that passes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--gate-config", default="configs/promotion_gate.yaml")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preferred-adapter", default=None, help="Adapter to test before numeric checkpoints.")
    parser.add_argument("--max-candidates", type=int, default=4, help="Number of numeric checkpoints to try.")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sentinel-manifest", default="data/manifests/sentinel_100_deleaked.json")
    parser.add_argument("--allow-missing-gradient-telemetry", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    candidates: list[Path] = []
    if args.preferred_adapter:
        preferred = Path(args.preferred_adapter)
        if preferred.exists():
            candidates.append(preferred)
    candidates.extend(path for path in _latest_checkpoints(checkpoint_dir, args.max_candidates) if path not in candidates)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not candidates:
        payload = {
            "eligible": False,
            "selected_checkpoint_path": None,
            "reason": "No adapter checkpoints found.",
            "attempts": [],
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("ERROR: no adapter checkpoints found", file=sys.stderr)
        return 1

    attempts: list[dict] = []
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = f"{project_root / 'src'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(project_root / "src")

    for candidate in candidates:
        gate_out = checkpoint_dir / "stage_gate_attempts" / candidate.stem
        cmd = [
            "bash",
            "tools/run_stage_gate.sh",
            "--config",
            args.config,
            "--adapter",
            str(candidate),
            "--stage",
            args.stage,
            "--gate-config",
            args.gate_config,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--num-samples",
            str(args.num_samples),
            "--sentinel-manifest",
            args.sentinel_manifest,
            "--out-dir",
            str(gate_out),
        ]
        if args.allow_missing_gradient_telemetry:
            cmd.append("--allow-missing-gradient-telemetry")

        print(f"Testing checkpoint candidate: {candidate}")
        result = subprocess.run(cmd, cwd=project_root, env=env)
        attempt = {
            "checkpoint_path": str(candidate),
            "gate_output_dir": str(gate_out),
            "exit_code": result.returncode,
            "passed": result.returncode == 0,
        }
        attempts.append(attempt)
        if result.returncode == 0:
            payload = {
                "eligible": True,
                "selected_checkpoint_path": str(candidate),
                "selected_adapter_sha256": _sha256(candidate),
                "selected_step": _checkpoint_sort_key(candidate) if _CHECKPOINT_RE.match(candidate.name) else None,
                "reason": "Latest candidate passing the full stage gate.",
                "attempts": attempts,
            }
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Selected adapter: {candidate}")
            return 0

    payload = {
        "eligible": False,
        "selected_checkpoint_path": None,
        "selected_adapter_sha256": None,
        "selected_step": None,
        "reason": "No candidate passed the full stage gate.",
        "attempts": attempts,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ERROR: no checkpoint passed the stage gate", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
