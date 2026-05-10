from __future__ import annotations

from .model_io import MlxVlmAdapter
from .normalize import normalize_tikz
from .prompting import build_compile_repair_prompt, extract_latex_from_response
from .schemas import CompileSummary, GenerationRequest
from .settings import PipelineConfig


class DatasetRepairer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.adapter = MlxVlmAdapter(config.model, config.memory)

    def repair_sample(self, code: str, summary: CompileSummary) -> str:
        prompt = build_compile_repair_prompt(code, summary)
        result = self.adapter.generate(
            GenerationRequest(
                description=prompt,
                max_tokens=self.config.model.max_output_tokens,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                top_k=self.config.model.top_k,
            )
        )
        return normalize_tikz(extract_latex_from_response(result.text))
