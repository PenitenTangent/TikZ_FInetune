from __future__ import annotations

from .base import EmbeddingMatrix


class EarthMoverReward:
    """Exact OT scorer using POT.

    This is intended for offline evaluation or reranking, not the hot inference loop.
    """

    def score_embeddings(self, reference: EmbeddingMatrix, candidate: EmbeddingMatrix) -> float:
        try:
            import numpy as np
            import ot
        except ImportError as exc:
            raise RuntimeError("POT and numpy are required for EMD scoring.") from exc

        ref = np.asarray(reference, dtype=float)
        cand = np.asarray(candidate, dtype=float)
        if ref.size == 0 or cand.size == 0:
            return 0.0

        ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
        cand = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        cosine = ref @ cand.T
        distance = np.clip(1.0 - cosine, 0.0, 2.0)
        ref_weights = np.full(ref.shape[0], 1.0 / ref.shape[0], dtype=float)
        cand_weights = np.full(cand.shape[0], 1.0 / cand.shape[0], dtype=float)

        cost = ot.emd2(ref_weights, cand_weights, distance)
        score = 1.0 - float(cost)
        return max(0.0, min(1.0, score))
