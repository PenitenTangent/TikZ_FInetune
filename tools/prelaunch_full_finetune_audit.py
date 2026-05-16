#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGE_CONFIGS = [ROOT / "configs" / f"curriculum_stage{i}.yaml" for i in range(1, 6)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _cache_len(path: Path) -> int:
    data = np.load(path, allow_pickle=True)
    return int(len(data))


def _check_stage_config(path: Path, failures: list[str]) -> None:
    config = _load_yaml(path)
    memory = config.get("memory", {})
    training = config.get("training", {})
    stage = path.stem.replace("curriculum_stage", "stage ")

    batch_size = int(memory.get("batch_size", 0))
    grad_accum = int(memory.get("gradient_accumulation_steps", 0))
    if batch_size != 1:
        failures.append(f"{stage}: memory.batch_size must be 1, got {batch_size}")
    if grad_accum <= 0:
        failures.append(f"{stage}: memory.gradient_accumulation_steps must be positive")

    dataset_path = ROOT / str(training.get("dataset_path", ""))
    cache_path = ROOT / str(training.get("pretokenized_cache_path", ""))
    if not dataset_path.exists():
        failures.append(f"{stage}: missing dataset {dataset_path.relative_to(ROOT)}")
        return

    rows = _jsonl_rows(dataset_path)
    expected_iters = ((rows + grad_accum - 1) // grad_accum) * grad_accum
    configured_iters = int(training.get("iters", 0))
    if configured_iters != expected_iters:
        failures.append(
            f"{stage}: training.iters={configured_iters}, expected rounded row count {expected_iters}"
        )

    if not cache_path.exists():
        failures.append(f"{stage}: missing pretokenized cache {cache_path.relative_to(ROOT)}")
        return

    cache_count = _cache_len(cache_path)
    if cache_count != rows:
        failures.append(f"{stage}: cache length {cache_count} does not match dataset rows {rows}")

    audit_path = cache_path.with_name(f"{cache_path.stem}_audit.json")
    if not audit_path.exists():
        failures.append(f"{stage}: missing cache audit {audit_path.relative_to(ROOT)}")
        return

    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"{stage}: unreadable cache audit {audit_path.relative_to(ROOT)}: {exc}")
        return

    recorded_sha = audit.get("source_jsonl_sha256")
    actual_sha = _sha256(dataset_path)
    if recorded_sha and recorded_sha != actual_sha:
        failures.append(f"{stage}: cache audit source_jsonl_sha256 mismatch")

    model_id = config.get("model", {}).get("model_id")
    cached_model_id = audit.get("model_id") or audit.get("tokenizer_id")
    if cached_model_id and cached_model_id != model_id:
        failures.append(f"{stage}: cache audit model/tokenizer mismatch")

    max_tokens = audit.get("max_tokens")
    model_config = config.get("model", {})
    config_max_tokens = model_config.get("max_context_tokens", model_config.get("max_output_tokens"))
    if max_tokens is not None and int(max_tokens) != int(config_max_tokens):
        failures.append(f"{stage}: cache audit max_tokens mismatch")


def _check_partition(failures: list[str]) -> None:
    stage_paths = []
    for config_path in STAGE_CONFIGS:
        training = _load_yaml(config_path).get("training", {})
        stage_paths.extend(["--stage", str(ROOT / str(training.get("dataset_path", "")))])
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "validate_curriculum_partition.py"), *stage_paths],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        failures.append("curriculum partition check failed:\n" + result.stdout.strip())


def _check_disk(failures: list[str], min_free_gib: float) -> None:
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / (1024**3)
    if free_gib < min_free_gib:
        failures.append(f"disk free space is {free_gib:.1f} GiB, below required {min_free_gib:.1f} GiB")


def _check_locks(failures: list[str]) -> None:
    for lock_path in (ROOT / "runs").glob("curriculum_stage*/train*.lock"):
        failures.append(f"stale run lock present: {lock_path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit launch readiness for the 5-stage full finetune.")
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    args = parser.parse_args()

    failures: list[str] = []
    for config_path in STAGE_CONFIGS:
        _check_stage_config(config_path, failures)
    _check_partition(failures)
    _check_disk(failures, args.min_free_gib)
    _check_locks(failures)

    if failures:
        print("FAIL: full finetune prelaunch audit found blockers:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("PASS: full finetune prelaunch audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
