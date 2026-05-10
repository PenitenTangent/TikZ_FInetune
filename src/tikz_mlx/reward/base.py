from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


EmbeddingMatrix = Sequence[Sequence[float]]


class EmbeddingEncoder(Protocol):
    def encode_image(self, image_path: str) -> EmbeddingMatrix:
        """Return patch embeddings for an image."""


class RewardBackend(Protocol):
    def score_embeddings(self, reference: EmbeddingMatrix, candidate: EmbeddingMatrix) -> float:
        """Score candidate embeddings against reference embeddings."""
