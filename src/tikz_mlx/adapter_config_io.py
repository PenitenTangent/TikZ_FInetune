"""Read, validate, and materialize mlx-vlm adapter configs for handoffs."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from .checkpointing import checkpoint_metadata_path, resolve_adapter_weights_path

GEMMA_E4B_6BIT_LORA_INPUT_DIM_REWRITES = {2400: 2560}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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
    return _read_json(cfg_path)


def load_adapter_lora_hyperparams_from_metadata(adapter_path: str | Path | None) -> dict[str, Any] | None:
    """Read LoRA hyperparameters from the checkpoint metadata sidecar if present."""
    weights_path = resolve_adapter_weights_path(adapter_path)
    if weights_path is None:
        return None
    payload = _read_json(checkpoint_metadata_path(weights_path))
    if payload is None:
        payload = _read_json(weights_path.parent / "checkpoint_metadata.json")
    if payload is None:
        return None
    resolved = payload.get("resolved_training_config")
    if not isinstance(resolved, dict):
        return None
    result: dict[str, Any] = {}
    if "lora_rank" in resolved:
        result["rank"] = resolved["lora_rank"]
    if "lora_alpha" in resolved:
        result["alpha"] = resolved["lora_alpha"]
    if "lora_dropout" in resolved:
        result["dropout"] = resolved["lora_dropout"]
    return result or None


def _coerce_lora_hyperparams(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    try:
        rank = int(payload["rank"])
        alpha = int(payload["alpha"])
        dropout = float(payload["dropout"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"rank": rank, "alpha": alpha, "dropout": dropout}


def load_source_lora_hyperparams(adapter_path: str | Path | None) -> dict[str, Any] | None:
    """Load source LoRA hyperparameters, preferring immutable checkpoint metadata.

    The root ``runs/adapter_config.json`` can be overwritten by later stages, so a
    checkpoint's metadata sidecar is a more reliable source when available.
    """
    return _coerce_lora_hyperparams(
        load_adapter_lora_hyperparams_from_metadata(adapter_path)
    ) or _coerce_lora_hyperparams(load_adapter_lora_hyperparams(adapter_path))


def infer_adapter_lora_rank(adapter_path: str | Path | None) -> int | None:
    """Infer LoRA rank from safetensors ``*.A``/``*.B`` tensor shapes.

    Returns ``None`` for missing or unreadable weights so plan/dry-run tests can
    still create a shim. Raises if readable LoRA tensors disagree on rank.
    """
    weights_path = resolve_adapter_weights_path(adapter_path)
    if weights_path is None or not weights_path.exists():
        return None
    try:
        from safetensors import safe_open
    except Exception:
        return None

    try:
        handle = safe_open(weights_path, framework="pt")
    except Exception:
        return None

    with handle:
        keys = set(handle.keys())
        ranks: set[int] = set()
        for key in keys:
            if not key.endswith(".A"):
                continue
            prefix = key[:-2]
            b_key = f"{prefix}.B"
            if b_key not in keys:
                continue
            a_shape = list(handle.get_slice(key).get_shape())
            b_shape = list(handle.get_slice(b_key).get_shape())
            if len(a_shape) != 2 or len(b_shape) != 2:
                continue
            if a_shape[1] != b_shape[0]:
                raise RuntimeError(
                    "LoRA tensor rank mismatch inside adapter: "
                    f"{key} shape={tuple(a_shape)} vs {b_key} shape={tuple(b_shape)}"
                )
            ranks.add(int(a_shape[1]))

    if not ranks:
        return None
    if len(ranks) != 1:
        raise RuntimeError(f"Adapter contains multiple LoRA ranks: {sorted(ranks)}")
    return next(iter(ranks))


def adapter_lora_input_dim_rewrites(
    adapter_path: str | Path | None,
    *,
    expected_model_id: str,
) -> dict[int, int]:
    """Return known-safe LoRA A input-dimension rewrites needed by an adapter."""
    if "gemma-4-e4b" not in expected_model_id.lower():
        return {}
    weights_path = resolve_adapter_weights_path(adapter_path)
    if weights_path is None or not weights_path.exists():
        return {}
    try:
        from safetensors import safe_open
    except Exception:
        return {}

    rewrites: dict[int, int] = {}
    with safe_open(weights_path, framework="pt") as handle:
        for key in handle.keys():
            if not key.endswith(".A"):
                continue
            shape = list(handle.get_slice(key).get_shape())
            if len(shape) != 2:
                continue
            source_dim = int(shape[0])
            target_dim = GEMMA_E4B_6BIT_LORA_INPUT_DIM_REWRITES.get(source_dim)
            if target_dim is not None:
                rewrites[source_dim] = target_dim
    return rewrites


def _write_target_adapter_config(
    adapter_dir: Path,
    *,
    target_rank: int,
    target_alpha: int,
    target_dropout: float,
) -> None:
    payload = {
        "rank": int(target_rank),
        "alpha": int(target_alpha),
        "dropout": float(target_dropout),
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def materialize_lora_handoff_adapter(
    *,
    source_adapter_path: str | Path,
    target_dir: str | Path,
    target_rank: int,
    target_alpha: int,
    target_dropout: float,
    seed: int = 17,
    init_std: float = 0.01,
    input_dim_rewrites: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Create an mlx-vlm adapter directory compatible with target LoRA settings.

    Upward rank changes are supported by preserving old dimensions, adding
    deterministic random ``A`` columns, and zero-initializing new ``B`` rows. If
    alpha changes, the copied ``B`` rows are rescaled so the effective adapter
    delta is unchanged at handoff.
    """
    source_weights = resolve_adapter_weights_path(source_adapter_path)
    if source_weights is None or not source_weights.exists():
        raise FileNotFoundError(f"Resume adapter weights do not exist: {source_adapter_path}")

    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_weights = target_dir / "adapters.safetensors"

    inferred_rank = infer_adapter_lora_rank(source_adapter_path)
    source_hparams = load_source_lora_hyperparams(source_adapter_path) or {}
    if inferred_rank is not None and source_hparams.get("rank") not in (None, inferred_rank):
        # A bare published adapter may sit next to a root adapter_config.json
        # already rewritten for the next stage. Trust tensor shape over that
        # mutable shim and fall back to alpha=2*rank to preserve the standard
        # LoRA scale used by this curriculum.
        source_hparams = {}
    source_rank = inferred_rank if inferred_rank is not None else source_hparams.get("rank")

    _write_target_adapter_config(
        target_dir,
        target_rank=target_rank,
        target_alpha=target_alpha,
        target_dropout=target_dropout,
    )

    result: dict[str, Any] = {
        "adapter_dir": str(target_dir),
        "source_weights": str(source_weights),
        "target_weights": str(target_weights),
        "source_rank": source_rank,
        "target_rank": int(target_rank),
        "source_alpha": source_hparams.get("alpha"),
        "target_alpha": int(target_alpha),
        "source_dropout": source_hparams.get("dropout"),
        "target_dropout": float(target_dropout),
        "expanded": False,
        "alpha_rescaled": False,
        "input_dim_rewritten": False,
        "linked_weights": False,
    }

    if source_rank is None:
        _link_or_copy(source_weights, target_weights)
        result["linked_weights"] = True
        result["warning"] = "Could not infer source LoRA rank; linked weights without expansion."
        return result

    source_rank = int(source_rank)
    if target_rank < source_rank:
        raise RuntimeError(
            "Cannot resume a higher-rank LoRA adapter into a lower-rank stage "
            f"(source rank={source_rank}, target rank={target_rank}). "
            "Use a same-or-higher rank stage or restart from base."
        )

    source_alpha = int(source_hparams.get("alpha", source_rank * 2))
    source_scale = source_alpha / source_rank
    target_scale = target_alpha / target_rank
    alpha_scale = source_scale / target_scale
    input_dim_rewrites = input_dim_rewrites or {}
    needs_weight_rewrite = (
        target_rank != source_rank
        or not math.isclose(alpha_scale, 1.0)
        or bool(input_dim_rewrites)
    )
    if not needs_weight_rewrite:
        _link_or_copy(source_weights, target_weights)
        result["linked_weights"] = True
        return result

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except Exception as exc:
        raise RuntimeError(
            "LoRA rank/alpha handoff requires torch and safetensors.torch to rewrite tensors."
        ) from exc

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensors: dict[str, Any] = {}
    metadata: dict[str, str] = {}
    with safe_open(source_weights, framework="pt") as handle:
        raw_metadata = handle.metadata()
        if raw_metadata:
            metadata.update({str(k): str(v) for k, v in raw_metadata.items()})
        keys = list(handle.keys())
        key_set = set(keys)
        for key in keys:
            tensor = handle.get_tensor(key)
            if key.endswith(".A") and f"{key[:-2]}.B" in key_set and tensor.ndim == 2:
                old_rank = int(tensor.shape[1])
                if old_rank != source_rank:
                    raise RuntimeError(
                        f"Unexpected LoRA rank for {key}: tensor rank={old_rank}, source rank={source_rank}"
                    )
                source_input_dim = int(tensor.shape[0])
                target_input_dim = int(input_dim_rewrites.get(source_input_dim, source_input_dim))
                expanded = torch.zeros(
                    (target_input_dim, target_rank),
                    dtype=tensor.dtype,
                    device="cpu",
                )
                expanded[:source_input_dim, :source_rank] = tensor
                extra = torch.randn(
                    (target_input_dim, target_rank - source_rank),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ) * init_std
                expanded[:, source_rank:] = extra.to(dtype=tensor.dtype)
                tensors[key] = expanded
            elif key.endswith(".B") and f"{key[:-2]}.A" in key_set and tensor.ndim == 2:
                old_rank = int(tensor.shape[0])
                if old_rank != source_rank:
                    raise RuntimeError(
                        f"Unexpected LoRA rank for {key}: tensor rank={old_rank}, source rank={source_rank}"
                    )
                expanded = torch.zeros(
                    (target_rank, tensor.shape[1]),
                    dtype=tensor.dtype,
                    device="cpu",
                )
                expanded[:source_rank, :] = tensor * alpha_scale
                tensors[key] = expanded
            else:
                tensors[key] = tensor

    metadata.update(
        {
            "lora_handoff_source_rank": str(source_rank),
            "lora_handoff_target_rank": str(target_rank),
            "lora_handoff_source_alpha": str(source_alpha),
            "lora_handoff_target_alpha": str(target_alpha),
            "lora_handoff_alpha_scale": f"{alpha_scale:.12g}",
        }
    )
    temp_weights = target_weights.with_name(f"{target_weights.name}.tmp")
    if temp_weights.exists():
        temp_weights.unlink()
    save_file(tensors, str(temp_weights), metadata=metadata)
    os.replace(temp_weights, target_weights)
    result["expanded"] = target_rank != source_rank
    result["alpha_rescaled"] = not math.isclose(alpha_scale, 1.0)
    result["input_dim_rewritten"] = bool(input_dim_rewrites)
    result["source_alpha"] = source_alpha
    return result


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


def validate_resumed_adapter_model_and_shape(
    *,
    adapter_path: str | Path | None,
    expected_model_id: str,
    expected_hidden_size: int | None = None,
    expected_lora_num_layers: int | None = None,
    expected_target_suffixes: tuple[str, ...] | None = None,
) -> None:
    """Verify that the resumed adapter belongs to the correct model and has compatible shapes.

    Enforces that the adapter was trained on the same base architecture to avoid
    immediate shape mismatches during forward passes.
    """
    if adapter_path in (None, ""):
        return
    weights_path = resolve_adapter_weights_path(adapter_path)
    if weights_path is None or not weights_path.exists():
        return

    # 1. Validate Model ID from metadata sidecar if available
    metadata_candidates = [
        checkpoint_metadata_path(weights_path),
        weights_path.parent / "checkpoint_metadata.json",
    ]
    for metadata_path in metadata_candidates:
        if not metadata_path.exists():
            continue
        metadata = _read_json(metadata_path)
        if metadata and "resolved_training_config" in metadata:
            resolved_config = metadata["resolved_training_config"]
            source_model_id = resolved_config.get("model_id")
            if source_model_id and source_model_id != expected_model_id:
                raise RuntimeError(
                    f"Resume adapter model mismatch: source={source_model_id}, target={expected_model_id}. "
                    "Restart from base or use a compatible checkpoint."
                )
            source_hidden_size = resolved_config.get("hidden_size")
            if source_hidden_size is not None and expected_hidden_size is not None:
                try:
                    source_hidden_size_int = int(source_hidden_size)
                except (TypeError, ValueError):
                    source_hidden_size_int = None
                if source_hidden_size_int is not None and source_hidden_size_int != expected_hidden_size:
                    raise RuntimeError(
                        "Resume adapter hidden size mismatch: "
                        f"source={source_hidden_size_int}, target={expected_hidden_size}. "
                        "Restart from base or use a compatible checkpoint."
                    )
            source_lora_num_layers = resolved_config.get("lora_num_layers")
            if source_lora_num_layers is not None and expected_lora_num_layers is not None:
                try:
                    source_lora_num_layers_int = int(source_lora_num_layers)
                except (TypeError, ValueError):
                    source_lora_num_layers_int = None
                if (
                    source_lora_num_layers_int is not None
                    and source_lora_num_layers_int != expected_lora_num_layers
                ):
                    raise RuntimeError(
                        "Resume adapter LoRA layer-count mismatch: "
                        f"source={source_lora_num_layers_int}, target={expected_lora_num_layers}. "
                        "Use a compatible checkpoint or materialize an explicit handoff."
                    )
            source_target_suffixes = resolved_config.get("target_suffixes")
            if source_target_suffixes is not None and expected_target_suffixes is not None:
                if list(source_target_suffixes) != list(expected_target_suffixes):
                    raise RuntimeError(
                        "Resume adapter LoRA target suffix mismatch: "
                        f"source={source_target_suffixes}, target={list(expected_target_suffixes)}. "
                        "Use a compatible checkpoint or restart from base."
                    )
            break

    # 2. Validate tensor shapes. Known-safe rewrites are materialized before
    # mlx-vlm loads the adapter, so allow those here and reject other stale dims.
    rewrites = adapter_lora_input_dim_rewrites(
        adapter_path,
        expected_model_id=expected_model_id,
    )
    try:
        from safetensors import safe_open
        with safe_open(weights_path, framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(".A"):
                    tensor_shape = handle.get_slice(key).get_shape()
                    in_dim = tensor_shape[0]
                    if in_dim in rewrites:
                        continue
                    if "gemma" in expected_model_id.lower() and in_dim == 2400:
                        raise RuntimeError(
                            f"Incompatible LoRA tensor shape detected in {key}: in_dim={in_dim}. "
                            "Expected 2560 for Gemma-4B. This adapter is likely stale or from a different architecture."
                        )
    except (ImportError, RuntimeError) as e:
        if isinstance(e, RuntimeError):
            raise e
        # If safetensors/torch missing, skip shape check but metadata check remains
        pass
