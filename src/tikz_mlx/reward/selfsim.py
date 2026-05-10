from __future__ import annotations

import math

from .base import EmbeddingMatrix


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class SelfSimReward:
    """Patch-level symmetric max-cosine scorer.

    This keeps the reward backend lightweight and pluggable. A future encoder can
    supply patch embeddings without changing the scoring interface.
    """

    def score_embeddings(self, reference: EmbeddingMatrix, candidate: EmbeddingMatrix) -> float:
        ref = [list(vector) for vector in reference]
        cand = [list(vector) for vector in candidate]
        if not ref or not cand:
            return 0.0

        forward = sum(max(_cosine_similarity(r, c) for c in cand) for r in ref) / len(ref)
        backward = sum(max(_cosine_similarity(c, r) for r in ref) for c in cand) / len(cand)
        score = (forward + backward) / 2.0
        return max(0.0, min(1.0, score))
