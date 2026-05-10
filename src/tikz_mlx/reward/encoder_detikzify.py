from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..model_io import clear_mlx_cache, configure_wired_limit
from ..mlx_runtime import import_mlx_core
from ..settings import MemoryConfig
from .base import EmbeddingMatrix


class RewardEncoderError(RuntimeError):
    """Raised when the frozen reward encoder cannot produce patch embeddings."""


@dataclass(slots=True)
class LoadedRewardEncoder:
    model: Any
    processor: Any


class FrozenDetikzifyEncoder:
    """Frozen image encoder wrapper for stage-2 reward scoring.

    The class is named after the intended DeTikZify-style reward encoder, but it
    accepts any MLX-converted vision-language model that exposes
    `model.encode_image(...)`.
    """

    def __init__(self, model_id: str, memory_config: MemoryConfig):
        self.model_id = model_id
        self.memory_config = memory_config
        self.loaded: LoadedRewardEncoder | None = None

    def _import_api(self) -> tuple[Any, Any]:
        try:
            import_mlx_core()
            from mlx_vlm import load
            from mlx_vlm.generate import prepare_inputs
        except ImportError as exc:
            raise RewardEncoderError(
                "mlx-vlm is required for the frozen stage-2 reward encoder."
            ) from exc
        return load, prepare_inputs

    def ensure_loaded(self) -> LoadedRewardEncoder:
        if self.loaded is not None:
            return self.loaded

        load, _ = self._import_api()
        configure_wired_limit(self.memory_config)
        model, processor = load(self.model_id)
        self.loaded = LoadedRewardEncoder(model=model, processor=processor)
        return self.loaded

    def unload(self) -> None:
        self.loaded = None
        clear_mlx_cache()

    def encode_image(self, image_path: str) -> EmbeddingMatrix:
        loaded = self.ensure_loaded()
        _, prepare_inputs = self._import_api()
        if not hasattr(loaded.model, "encode_image"):
            raise RewardEncoderError(
                f"Reward model `{self.model_id}` does not expose `encode_image`."
            )

        add_special_tokens = (
            getattr(loaded.processor, "chat_template", None) is None
            if loaded.model.config.model_type in {"gemma3", "gemma3n", "gemma4"}
            else True
        )
        inputs = prepare_inputs(
            loaded.processor,
            images=[image_path],
            prompts="",
            image_token_index=getattr(loaded.model.config, "image_token_index", None),
            add_special_tokens=add_special_tokens,
            return_tensors="mlx",
        )
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RewardEncoderError(
                f"Reward model `{self.model_id}` could not prepare image inputs for `{image_path}`."
            )

        features = loaded.model.encode_image(pixel_values)
        matrix = self._coerce_embedding_matrix(features)
        if not matrix:
            raise RewardEncoderError(
                f"Reward model `{self.model_id}` returned empty image embeddings for `{image_path}`."
            )
        return matrix

    def _coerce_embedding_matrix(self, features: Any) -> EmbeddingMatrix:
        candidate = self._unwrap_features(features)
        if hasattr(candidate, "tolist"):
            candidate = candidate.tolist()
        if not isinstance(candidate, list):
            raise RewardEncoderError("Reward encoder returned an unsupported embedding container.")

        while candidate and isinstance(candidate[0], list) and candidate[0] and isinstance(candidate[0][0], list):
            candidate = candidate[0]

        matrix: list[list[float]] = []
        for row in candidate:
            if isinstance(row, list):
                matrix.append([float(value) for value in row])
        return matrix

    @staticmethod
    def _unwrap_features(features: Any) -> Any:
        candidate = features
        if isinstance(candidate, dict):
            for key in ("last_hidden_state", "hidden_states", "image_embeds", "embeddings"):
                if key in candidate:
                    candidate = candidate[key]
                    break

        for attribute in ("last_hidden_state", "image_embeds"):
            if hasattr(candidate, attribute):
                return getattr(candidate, attribute)

        hidden_states = getattr(candidate, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]

        if isinstance(candidate, (tuple, list)) and candidate:
            return FrozenDetikzifyEncoder._unwrap_features(candidate[0])

        return candidate
