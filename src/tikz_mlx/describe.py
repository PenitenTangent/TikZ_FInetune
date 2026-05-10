from __future__ import annotations

from .model_io import MlxVlmAdapter
from .prompting import DESCRIPTION_PROMPT, DESCRIPTION_REQUEST
from .schemas import GenerationRequest
from .settings import PipelineConfig


class FigureDescriber:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.adapter = MlxVlmAdapter(config.model, config.memory)

    def describe_image(self, image_path: str) -> str:
        result = self.adapter.generate(
            GenerationRequest(
                description=DESCRIPTION_REQUEST,
                image_paths=[image_path],
                system_prompt=DESCRIPTION_PROMPT,
                max_tokens=self.config.model.max_output_tokens,
                temperature=0.2,
                top_p=0.95,
                top_k=self.config.model.top_k,
            )
        )
        return result.text.strip()
