from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_NAMED_CHECKPOINTS: tuple[str, ...] = (
    "policy_init",
    "last",
    "last_prev",
    "last_probe_pass",
    "best_by_eval",
    "last_epoch_boundary",
)


@dataclass(slots=True)
class CheckpointContext:
    epoch: int | None = None
    global_step: int | None = None
    batch_in_epoch: int | None = None
    sample_cursor_in_epoch: int | None = None
    dataset_snapshot_id: str | None = None
    epoch_order_checksum: str | None = None
    training_config_fingerprint: str | None = None


def utc_now_iso8601() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def extract_iteration_from_checkpoint_name(checkpoint_path: Path) -> int | None:
    match = re.match(r"^(\d+)_adapters\.safetensors$", checkpoint_path.name)
    if not match:
        return None
    return int(match.group(1))


def extract_layer_index(module_name: str) -> int | None:
    """Extract layer index from a module name like 'model.layers.13.per_layer_projection'."""
    parts = module_name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def get_parent_module(root: Any, module_name: str) -> tuple[Any, str]:
    """Navigate to the parent of a module by name.

    Returns (parent_module, leaf_name).
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def unwrap_lora_layer(root: Any, module_name: str, base_layer: Any) -> None:
    """Replace a LoRA layer with its base linear layer, handling both attributes and indices."""
    parent, leaf_name = get_parent_module(root, module_name)
    if leaf_name.isdigit():
        parent[int(leaf_name)] = base_layer
    else:
        setattr(parent, leaf_name, base_layer)


def checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp_path, path)


def link_or_copy_atomic(source: Path, target: Path) -> None:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint source does not exist: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f"{target.name}.tmp")
    if temp_target.exists() or temp_target.is_symlink():
        temp_target.unlink()

    try:
        os.link(source, temp_target)
    except OSError:
        shutil.copy2(source, temp_target)
    os.replace(temp_target, target)


def _sanitize_metrics(metrics: dict[str, float] | None) -> dict[str, float]:
    if not metrics:
        return {}

    sanitized: dict[str, float] = {}
    for key, value in metrics.items():
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        sanitized[str(key)] = numeric_value
    return sanitized


def build_canonical_checkpoint_metadata(
    *,
    stage: str,
    run_id: str,
    checkpoint_role: str,
    checkpoint_path: Path,
    context: CheckpointContext,
    source_checkpoint_path: Path | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso8601(),
        "stage": stage,
        "run_id": run_id,
        "checkpoint_role": checkpoint_role,
        "checkpoint_path": str(checkpoint_path.expanduser().resolve()),
        "source_checkpoint_path": (
            str(source_checkpoint_path.expanduser().resolve())
            if source_checkpoint_path is not None
            else None
        ),
        "model_state": str(checkpoint_path.expanduser().resolve()),
        "optimizer_state": None,
        "scheduler_state": None,
        "scaler_state": None,
        "epoch": context.epoch,
        "global_step": context.global_step,
        "batch_in_epoch": context.batch_in_epoch,
        "sample_cursor_in_epoch": context.sample_cursor_in_epoch,
        "dataset_snapshot_id": context.dataset_snapshot_id,
        "epoch_order_checksum": context.epoch_order_checksum,
        "training_config_fingerprint": context.training_config_fingerprint,
        "rng_states": {
            "python": None,
            "numpy": None,
            "mlx": None,
        },
        "metrics": _sanitize_metrics(metrics),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_checkpoint_metadata(checkpoint_path: Path, payload: dict[str, Any]) -> Path:
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    write_json_atomic(metadata_path, payload)
    return metadata_path


def resolve_adapter_weights_path(adapter_path: str | Path | None) -> Path | None:
    if adapter_path in (None, ""):
        return None
    resolved = Path(adapter_path).expanduser().resolve()
    if resolved.is_dir():
        weights_path = resolved / "adapters.safetensors"
        return weights_path if weights_path.exists() else None
    return resolved if resolved.exists() else None


class NamedCheckpointPolicyManager:
    def __init__(
        self,
        *,
        named_dir: Path,
        stage: str,
        run_id: str,
        include_reward_spike: bool = False,
    ) -> None:
        self.named_dir = named_dir.expanduser().resolve()
        self.named_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.run_id = run_id
        self.include_reward_spike = include_reward_spike
        self._best_metric_value: float | None = None

    def record_source_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_role: str,
        context: CheckpointContext,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        metadata = build_canonical_checkpoint_metadata(
            stage=self.stage,
            run_id=self.run_id,
            checkpoint_role=checkpoint_role,
            checkpoint_path=checkpoint_path,
            source_checkpoint_path=checkpoint_path,
            context=context,
            metrics=metrics,
            extra=extra,
        )
        return write_checkpoint_metadata(checkpoint_path, metadata)

    def ensure_policy_init(
        self,
        *,
        source_checkpoint_path: Path,
        context: CheckpointContext,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        alias_path = self._alias_path("policy_init")
        if alias_path.exists():
            return alias_path
        return self._set_alias(
            "policy_init",
            source_checkpoint_path=source_checkpoint_path,
            context=context,
            metrics=metrics,
            immutable=True,
        )

    def update_last(
        self,
        *,
        source_checkpoint_path: Path,
        context: CheckpointContext,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        current_last = self._alias_path("last")
        if current_last.exists():
            self._set_alias(
                "last_prev",
                source_checkpoint_path=current_last,
                context=context,
                metrics=metrics,
                extra={"chained_from": "last"},
            )
        return self._set_alias(
            "last",
            source_checkpoint_path=source_checkpoint_path,
            context=context,
            metrics=metrics,
        )

    def update_best_by_eval(
        self,
        *,
        source_checkpoint_path: Path,
        metric_name: str,
        metric_value: float,
        higher_is_better: bool,
        context: CheckpointContext,
        metrics: dict[str, float] | None = None,
    ) -> Path | None:
        value = float(metric_value)
        if not math.isfinite(value):
            return None

        if self._best_metric_value is None:
            improved = True
        elif higher_is_better:
            improved = value > self._best_metric_value
        else:
            improved = value < self._best_metric_value

        if not improved:
            return None

        self._best_metric_value = value
        return self._set_alias(
            "best_by_eval",
            source_checkpoint_path=source_checkpoint_path,
            context=context,
            metrics=metrics,
            extra={
                "best_metric_name": metric_name,
                "best_metric_value": value,
                "best_metric_higher_is_better": higher_is_better,
            },
        )

    def update_last_epoch_boundary(
        self,
        *,
        source_checkpoint_path: Path,
        context: CheckpointContext,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        return self._set_alias(
            "last_epoch_boundary",
            source_checkpoint_path=source_checkpoint_path,
            context=context,
            metrics=metrics,
        )

    def update_last_probe_pass(
        self,
        *,
        source_checkpoint_path: Path,
        context: CheckpointContext,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        return self._set_alias(
            "last_probe_pass",
            source_checkpoint_path=source_checkpoint_path,
            context=context,
            metrics=metrics,
            extra=extra,
        )

    def update_last_pre_reward_spike(
        self,
        *,
        source_checkpoint_path: Path,
        context: CheckpointContext,
        metrics: dict[str, float] | None = None,
    ) -> Path | None:
        if not self.include_reward_spike:
            return None
        return self._set_alias(
            "last_pre_reward_spike",
            source_checkpoint_path=source_checkpoint_path,
            context=context,
            metrics=metrics,
        )

    def _set_alias(
        self,
        alias: str,
        *,
        source_checkpoint_path: Path,
        context: CheckpointContext,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
        immutable: bool = False,
    ) -> Path:
        if alias not in BASE_NAMED_CHECKPOINTS and alias != "last_pre_reward_spike":
            raise ValueError(f"Unsupported checkpoint alias: {alias}")
        if alias == "last_pre_reward_spike" and not self.include_reward_spike:
            raise ValueError("last_pre_reward_spike alias is disabled for this manager.")

        alias_path = self._alias_path(alias)
        if immutable and alias_path.exists():
            return alias_path

        link_or_copy_atomic(source_checkpoint_path, alias_path)
        metadata = build_canonical_checkpoint_metadata(
            stage=self.stage,
            run_id=self.run_id,
            checkpoint_role=alias,
            checkpoint_path=alias_path,
            source_checkpoint_path=source_checkpoint_path,
            context=context,
            metrics=metrics,
            extra=extra,
        )
        write_checkpoint_metadata(alias_path, metadata)
        return alias_path

    def _alias_path(self, alias: str) -> Path:
        return self.named_dir / f"{alias}.safetensors"
