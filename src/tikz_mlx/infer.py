from __future__ import annotations

from dataclasses import replace

from .model_io import MlxVlmAdapter
from .normalize import normalize_tikz
from .prompting import (
    build_compile_repair_prompt,
    build_generation_prompt,
    build_visual_repair_prompt,
    extract_latex_from_response,
)
from .schemas import CompileSummary, GenerationRequest, RepairMode, RefinementAttempt
from .settings import DecodingConfig, PipelineConfig

# Temperature step applied per compile-repair round to escape degenerate modes.
# Capped at ADAPTIVE_TEMPERATURE_MAX to avoid pure noise (plan Extended §5).
_ADAPTIVE_TEMPERATURE_STEP = 0.15
_ADAPTIVE_TEMPERATURE_MAX = 0.90


def _bump_temperature(decoding: DecodingConfig, *, rounds: int) -> DecodingConfig:
    """Return a copy of *decoding* with temperature raised by *rounds* steps.

    If the base temperature is None (greedy), treats it as 0.0 before bumping.
    The result is clamped to _ADAPTIVE_TEMPERATURE_MAX.
    """
    if rounds <= 0:
        return decoding
    base = decoding.temperature if decoding.temperature is not None else 0.0
    new_temp = min(_ADAPTIVE_TEMPERATURE_MAX, base + rounds * _ADAPTIVE_TEMPERATURE_STEP)
    return replace(decoding, temperature=new_temp)


class InferenceEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.adapter = MlxVlmAdapter(config.model, config.memory)

    def generate_initial(self, description: str) -> RefinementAttempt:
        prompt = build_generation_prompt(description)
        code = self._generate_code(prompt, [], self.config.inference.initial_decoding)
        return RefinementAttempt(
            mode=RepairMode.INITIAL_GENERATION,
            prompt=prompt,
            generated_code=code,
        )

    def repair_compile_failure(
        self,
        code: str,
        summary: CompileSummary,
        *,
        repair_round: int = 1,
    ) -> RefinementAttempt:
        """Generate a repaired version of *code* using the compiler error summary.

        Args:
            code: The TikZ code that failed to compile.
            summary: Compiler output containing key errors and line hints.
            repair_round: 1-indexed repair attempt count. Temperature increases
                by _ADAPTIVE_TEMPERATURE_STEP per round (plan Extended §5).
        """
        prompt = build_compile_repair_prompt(code, summary)
        # Adaptive temperature: higher temperature on each subsequent repair round
        # to escape the local degenerate mode that caused the previous failure.
        decoding = _bump_temperature(
            self.config.inference.compile_repair_decoding,
            rounds=repair_round - 1,
        )
        repaired = self._generate_code(prompt, [], decoding)
        return RefinementAttempt(
            mode=RepairMode.COMPILE_REPAIR,
            prompt=prompt,
            generated_code=repaired,
        )

    def repair_visual_output(self, description: str, code: str, debug_image_path: str) -> RefinementAttempt:
        prompt = build_visual_repair_prompt(description, code)
        repaired = self._generate_code(
            prompt,
            [debug_image_path],
            self.config.inference.visual_repair_decoding,
        )
        return RefinementAttempt(
            mode=RepairMode.VISUAL_REPAIR,
            prompt=prompt,
            generated_code=repaired,
        )

    def _generate_code(self, prompt: str, image_paths: list[str], decoding: DecodingConfig) -> str:
        result = self.adapter.generate(
            GenerationRequest(
                description=prompt,
                image_paths=image_paths,
                max_tokens=decoding.max_tokens,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                top_k=decoding.top_k,
                min_p=decoding.min_p,
                repetition_penalty=decoding.repetition_penalty,
            )
        )
        return normalize_tikz(extract_latex_from_response(result.text))
