#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


DEFAULT_ADAPTERS = [
    "runs/tikz_stage1_adapter.safetensors",
    "runs/tikz_stage2_adapter.safetensors",
    "runs/tikz_stage3_adapter.safetensors",
    "runs/tikz_stage4_adapter.safetensors",
    "runs/tikz_stage5_adapter.safetensors",
    "runs/curriculum_stage4/final_adapter.safetensors",
    "runs/curriculum_stage5/final_adapter.safetensors",
]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_adapter(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return payload

    try:
        from safetensors import safe_open
    except Exception as exc:
        payload["error"] = f"safetensors import failed: {exc}"
        return payload

    payload["sha256"] = sha256(path)
    tensor_count = 0
    elem_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    max_abs = 0.0
    first_keys: list[str] = []

    last_error: Exception | None = None
    handle = None
    for framework in ("pt", "np"):
        try:
            handle = safe_open(path, framework=framework)
            break
        except Exception as exc:
            last_error = exc
    if handle is None:
        payload["error"] = f"could not open safetensors: {last_error}"
        return payload

    with handle:
        keys = list(handle.keys())
        first_keys = keys[:8]
        for key in keys:
            arr = handle.get_tensor(key)
            if hasattr(arr, "detach"):
                arr = arr.detach().float().cpu()
            tensor_count += 1
            arr_size = int(arr.numel()) if hasattr(arr, "numel") else int(arr.size)
            elem_count += arr_size
            abs_sum += float(abs(arr).sum())
            sq_sum += float((arr * arr).sum())
            max_abs = max(max_abs, float(abs(arr).max()) if arr_size else 0.0)

    payload.update(
        {
            "tensor_count": tensor_count,
            "first_keys": first_keys,
            "elements": elem_count,
            "mean_abs": abs_sum / elem_count if elem_count else 0.0,
            "max_abs": max_abs,
            "l2_norm": math.sqrt(sq_sum),
        }
    )
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if metadata_path.exists():
        payload["metadata_path"] = str(metadata_path)
        try:
            payload["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload["metadata_error"] = "metadata is not valid JSON"
    return payload


def inspect_stage_manifest(stage: int, *, root: Path) -> dict[str, Any]:
    config_path = root / "configs" / f"curriculum_stage{stage}.yaml"
    adapter_path = root / "runs" / f"tikz_stage{stage}_adapter.safetensors"
    payload: dict[str, Any] = {
        "stage": stage,
        "config_path": str(config_path),
        "adapter_path": str(adapter_path),
        "config_exists": config_path.exists(),
        "adapter_exists": adapter_path.exists(),
    }
    if config_path.exists():
        payload["config_sha256"] = sha256(config_path)
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        training = cfg.get("training") or {}
        model = cfg.get("model") or {}
        payload.update(
            {
                "base_model_id": model.get("model_id"),
                "lora_rank": training.get("lora_rank"),
                "lora_alpha": training.get("lora_alpha"),
                "lora_dropout": training.get("lora_dropout"),
                "lora_num_layers": training.get("lora_num_layers"),
                "dataset_path": training.get("dataset_path"),
                "pretokenized_cache_path": training.get("pretokenized_cache_path"),
            }
        )
        for key in ("dataset_path", "pretokenized_cache_path"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                path = Path(value)
                if not path.is_absolute():
                    path = root / path
                payload[f"{key}_exists"] = path.exists()
                if path.exists():
                    payload[f"{key}_sha256"] = sha256(path)
    if adapter_path.exists():
        payload["adapter_sha256"] = sha256(adapter_path)
    adapter_config = root / "runs" / "adapter_config.json"
    payload["adapter_config_path"] = str(adapter_config)
    payload["adapter_config_exists"] = adapter_config.exists()
    if adapter_config.exists():
        payload["adapter_config_sha256"] = sha256(adapter_config)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapter health and lineage inspection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect")
    inspect_cmd.add_argument("paths", nargs="*", default=DEFAULT_ADAPTERS)
    inspect_cmd.add_argument("--output")

    manifest_cmd = subparsers.add_parser("stage-manifests")
    manifest_cmd.add_argument("--stages", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    manifest_cmd.add_argument("--output")

    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "inspect":
        payload = [inspect_adapter(Path(path)) for path in args.paths]
    else:
        payload = [inspect_stage_manifest(stage, root=root) for stage in args.stages]

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
