"""Read and validate mlx-vlm adapter_config.json for curriculum handoffs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .checkpointing import resolve_adapter_weights_path


def resolve_adapter_config_path(adapter_path: str | Path | None) -> Path | None:
    """Return adapter_config.json path for a checkpoint dir, weights file, or shim."""
    if adapter_path in (None, ""):
        return None
    path = Path(adapter_path).expanduser().resolve()
    if path.is_dir():
        candidate = path / "adapter_config.json"
        return candidate if candidate.exists() else None
    parent = path.parent
    if (parent / "adapter_config.json").exists():
        return parent / "adapter_config.json"
    if (parent.parent / "adapter_config.json").exists():
        return parent.parent / "adapter_config.json"
    return None


def load_adapter_lora_hyperparams(adapter_path: str | Path | None) -> dict[str, Any] | None:
    """Read the LoRA hyperparameters (rank, alpha, dropout) from adapter_config.json.

    Returns None if the config file is missing or unreadable.
    """
    cfg_path = resolve_adapter_config_path(adapter_path)
    if cfg_path is None:
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_resumed_adapter_lora_hyperparams(
    *,
    adapter_path: str | Path | None,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    """Verify that a resumed checkpoint has LoRA settings compatible with the current run.

    Prevents weight misalignment during curriculum handoffs by enforcing strict 
    hyperparameter consistency.
    """
    if adapter_path in (None, ""):
        return
    resolved_weights = resolve_adapter_weights_path(adapter_path)
    if resolved_weights is None or not resolved_weights.exists():
        return

    payload = load_adapter_lora_hyperparams(adapter_path)
    if payload is None:
        raise RuntimeError(
            "Resume adapter is missing a readable adapter_config.json with LoRA settings. "
            f"Checked near: {adapter_path!s}"
        )

    def _as_int(key: str) -> int | None:
        raw = payload.get(key)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _as_float(key: str) -> float | None:
        raw = payload.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    # mlx-vlm stores keys: rank, alpha, dropout (see mlx_vlm.trainer.utils.get_peft_model).
    ar = _as_int("rank")
    aa = _as_int("alpha")
    ad = _as_float("dropout")
    if None in (ar, aa, ad):
        raise RuntimeError(
            "adapter_config.json must include numeric rank, alpha, and dropout for resume validation. "
            f"Got keys={sorted(payload.keys())}"
        )

    if ar != lora_rank or aa != lora_alpha or not math.isclose(ad, lora_dropout, rel_tol=1e-5, abs_tol=1e-8):
        raise RuntimeError(
            "Resume adapter LoRA hyperparameters do not match training config — "
            f"adapter(rank={ar}, alpha={aa}, dropout={ad}) vs "
            f"training(lora_rank={lora_rank}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}). "
            "Weights would be misaligned; use a compatible checkpoint or regenerate the adapter."
        )
