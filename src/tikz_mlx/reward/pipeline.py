from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..compiler import CompilerService
from ..debug_render import (
    RasterizationError,
    load_render_config,
    prepare_image_for_reward_encoder,
    rasterize_pdf,
)
from ..schema_validation import is_canonical_tikz_document
from ..schemas import CompileStatus, CompileSummary, Stage2Sample
from ..settings import PipelineConfig
from .base import EmbeddingEncoder, EmbeddingMatrix, RewardBackend
from .emd import EarthMoverReward
from .selfsim import SelfSimReward


def build_reward_backend(name: str) -> RewardBackend:
    if name == "emd":
        return EarthMoverReward()
    if name == "selfsim":
        return SelfSimReward()
    if name == "gemini":
        from .gemini import get_gemini_reward
        # We return a wrapper that matches the RewardBackend protocol but expects paths
        return get_gemini_reward()
    raise ValueError(f"Unsupported reward backend: {name}")


@dataclass(slots=True)
class Stage2RewardResult:
    reward: float
    compiled: bool
    format_ok: bool
    compile_summary: CompileSummary | None = None
    reference_image_path: Path | None = None
    candidate_image_path: Path | None = None


class Stage2RewardPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        encoder: EmbeddingEncoder,
        backend: RewardBackend | None = None,
    ):
        self.config = config
        self.encoder = encoder
        self.backend = backend or build_reward_backend(config.training.stage2.reward_backend)
        self.is_api_reward = config.training.stage2.reward_backend == "gemini"
        self.compiler = CompilerService(config.compiler)
        self.render_config = load_render_config(config.paths.root_dir)
        self.render_config_fingerprint = hashlib.sha256(
            self.render_config.path.read_bytes()
        ).hexdigest()[:12]

    def score_candidate(
        self,
        sample: Stage2Sample,
        candidate_code: str,
        output_dir: str | Path,
    ) -> Stage2RewardResult:
        if not self.has_required_document_format(candidate_code):
            return Stage2RewardResult(reward=0.0, compiled=False, format_ok=False)

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        summary = self.compiler.compile_document(candidate_code, output_dir=target_dir, job_name="candidate")
        if summary.status != CompileStatus.SUCCESS or summary.pdf_path is None:
            return Stage2RewardResult(
                reward=0.0,
                compiled=False,
                format_ok=True,
                compile_summary=summary,
            )

        try:
            candidate_image = rasterize_pdf(summary.pdf_path, target_dir, render_config=self.render_config)
            reference_image = self._resolve_reference_image(sample)
        except (RasterizationError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return Stage2RewardResult(
                reward=0.0,
                compiled=True,
                format_ok=True,
                compile_summary=summary,
            )

        if self.is_api_reward:
            # Call Gemini directly with image paths
            reward = self.backend.score_images(reference_image, candidate_image)
        else:
            reference_embeddings = self._resolve_reference_embeddings(sample, reference_image)
            if reference_embeddings is None:
                return Stage2RewardResult(
                    reward=0.0,
                    compiled=True,
                    format_ok=True,
                    compile_summary=summary,
                    candidate_image_path=candidate_image,
                )

            candidate_embeddings = self.encoder.encode_image(
                str(prepare_image_for_reward_encoder(candidate_image, self.render_config))
            )
            reward = self.backend.score_embeddings(reference_embeddings, candidate_embeddings)
        
        return Stage2RewardResult(
            reward=reward * 5.0,
            compiled=True,
            format_ok=True,
            compile_summary=summary,
            reference_image_path=reference_image,
            candidate_image_path=candidate_image,
        )

    @staticmethod
    def has_required_document_format(code: str) -> bool:
        return is_canonical_tikz_document(code)

    def _resolve_reference_embeddings(
        self,
        sample: Stage2Sample,
        reference_image: Path | None,
    ) -> EmbeddingMatrix | None:
        configured_path = sample.reference_embedding_path
        if configured_path:
            embedding_path = Path(configured_path).expanduser()
            if embedding_path.exists():
                return self._load_embeddings(embedding_path)

        cache_path = (
            self.config.training.stage2.reward_cache_dir
            / self.render_config_fingerprint
            / f"{sample.sample_id}.json"
        )
        if self.config.training.stage2.cache_reference_artifacts and cache_path.exists():
            return self._load_embeddings(cache_path)

        if reference_image is None:
            return None

        embeddings = self.encoder.encode_image(str(reference_image))
        if self.config.training.stage2.cache_reference_artifacts:
            self._save_embeddings(cache_path, embeddings)
        return embeddings

    def _resolve_reference_image(self, sample: Stage2Sample) -> Path | None:
        if sample.reference_image_path:
            image_path = Path(sample.reference_image_path).expanduser()
            return image_path if image_path.exists() else None
        if not sample.reference_code:
            return None

        sample_cache_dir = (
            self.config.training.stage2.reward_cache_dir / self.render_config_fingerprint / sample.sample_id
        )
        reference_png = sample_cache_dir / "reference.png"
        if self.config.training.stage2.cache_reference_artifacts and reference_png.exists():
            return reference_png

        sample_cache_dir.mkdir(parents=True, exist_ok=True)
        summary = self.compiler.compile_document(
            sample.reference_code,
            output_dir=sample_cache_dir / "compile",
            job_name="reference",
        )
        if summary.status != CompileStatus.SUCCESS or summary.pdf_path is None:
            return None
        return rasterize_pdf(summary.pdf_path, sample_cache_dir, render_config=self.render_config)

    @staticmethod
    def _load_embeddings(path: Path) -> EmbeddingMatrix:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [[float(value) for value in row] for row in data]

    @staticmethod
    def _save_embeddings(path: Path, embeddings: EmbeddingMatrix) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump([[float(value) for value in row] for row in embeddings], handle)
