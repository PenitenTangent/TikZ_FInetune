import json
from pathlib import Path

from tikz_mlx.reward.emd import EarthMoverReward
from tikz_mlx.reward.pipeline import Stage2RewardPipeline
from tikz_mlx.reward.selfsim import SelfSimReward
from tikz_mlx.schemas import CompileStatus, CompileSummary, Stage2Sample
from tikz_mlx.settings import load_config
from PIL import Image

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lora_prod.yaml"


class _DummyEncoder:
    def encode_image(self, image_path: str):
        return [[0.0, 1.0]]


class _DummyBackend:
    def __init__(self):
        self.calls = []

    def score_embeddings(self, reference, candidate):
        self.calls.append((reference, candidate))
        return 0.75


class _FakeCompiler:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path

    def compile_document(self, latex_source: str, output_dir, job_name: str = "document"):
        return CompileSummary(
            status=CompileStatus.SUCCESS,
            return_code=0,
            key_errors=[],
            line_hints=[],
            missing_packages=[],
            stdout="",
            stderr="",
            log_text="",
            elapsed_seconds=0.1,
            pdf_path=self.pdf_path,
        )


def test_selfsim_reward_scores_identical_embeddings_as_one() -> None:
    reward = SelfSimReward()
    score = reward.score_embeddings([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert score == 1.0


def test_emd_reward_scores_identical_embeddings_as_one() -> None:
    reward = EarthMoverReward()
    score = reward.score_embeddings([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert score == 1.0


def test_emd_reward_returns_zero_for_empty_embeddings() -> None:
    reward = EarthMoverReward()
    score = reward.score_embeddings([], [[1.0, 0.0]])
    assert score == 0.0


def test_stage2_reward_pipeline_requires_document_format() -> None:
    config = load_config(CONFIG_PATH)
    pipeline = Stage2RewardPipeline(config, encoder=_DummyEncoder(), backend=_DummyBackend())
    sample = Stage2Sample(sample_id="sample-1", prompt_text="prompt")
    result = pipeline.score_candidate(sample, r"\begin{document}\end{document}", output_dir="unused")
    assert result.reward == 0.0
    assert result.format_ok is False
    assert result.compiled is False


def test_stage2_reward_pipeline_rejects_markdown_fences() -> None:
    config = load_config(CONFIG_PATH)
    pipeline = Stage2RewardPipeline(config, encoder=_DummyEncoder(), backend=_DummyBackend())
    sample = Stage2Sample(sample_id="sample-fence", prompt_text="prompt")
    candidate = """\\documentclass[tikz]{standalone}
\\begin{document}
```latex
\\begin{tikzpicture}
\\end{tikzpicture}
```
\\end{document}"""
    result = pipeline.score_candidate(sample, candidate, output_dir="unused")
    assert result.reward == 0.0
    assert result.format_ok is False
    assert result.compiled is False


def test_stage2_reward_pipeline_rejects_duplicated_wrappers() -> None:
    config = load_config(CONFIG_PATH)
    pipeline = Stage2RewardPipeline(config, encoder=_DummyEncoder(), backend=_DummyBackend())
    sample = Stage2Sample(sample_id="sample-dup", prompt_text="prompt")
    candidate = """\\documentclass[tikz]{standalone}
\\begin{document}
\\documentclass[tikz]{standalone}
\\begin{document}
\\end{document}
\\end{document}"""
    result = pipeline.score_candidate(sample, candidate, output_dir="unused")
    assert result.reward == 0.0
    assert result.format_ok is False
    assert result.compiled is False


def test_stage2_reward_pipeline_uses_reference_embeddings(monkeypatch, tmp_path) -> None:
    config = load_config(CONFIG_PATH)
    backend = _DummyBackend()
    pipeline = Stage2RewardPipeline(config, encoder=_DummyEncoder(), backend=backend)

    candidate_pdf = tmp_path / "candidate.pdf"
    candidate_pdf.write_text("pdf", encoding="utf-8")
    candidate_png = tmp_path / "candidate.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(candidate_png)
    pipeline.compiler = _FakeCompiler(candidate_pdf)
    monkeypatch.setattr(
        "tikz_mlx.reward.pipeline.rasterize_pdf",
        lambda pdf_path, output_dir, render_config=None: candidate_png,
    )

    reference_embedding_path = tmp_path / "reference_embeddings.json"
    reference_embedding_path.write_text(json.dumps([[1.0, 0.0]]), encoding="utf-8")
    sample = Stage2Sample(
        sample_id="sample-2",
        prompt_text="prompt",
        reference_embedding_path=str(reference_embedding_path),
    )
    result = pipeline.score_candidate(
        sample,
        "\\documentclass[tikz]{standalone}\n\\begin{document}\n\\end{document}",
        output_dir=tmp_path / "run",
    )
    assert result.reward == 3.75
    assert result.compiled is True
    assert result.format_ok is True
    assert backend.calls == [([[1.0, 0.0]], [[0.0, 1.0]])]
