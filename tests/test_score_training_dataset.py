from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tikz_mlx.prompting import build_generation_prompt
from tikz_mlx.settings import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lora_prod.yaml"


def _load_score_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "score_training_dataset.py"
    spec = importlib.util.spec_from_file_location("score_training_dataset", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load score_training_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(description: str) -> dict:
    return {
        "sample_id": "sample-1",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": build_generation_prompt(description)}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "```latex\n\\begin{tikzpicture}\\end{tikzpicture}\n```"}],
            },
        ],
        "metadata": {},
    }


def test_extract_description_from_generation_prompt() -> None:
    module = _load_score_module()
    record = _record("Draw a red triangle.")

    assert module._extract_description(record) == "Draw a red triangle."


def test_apply_alignment_scores_sets_sample_weight(monkeypatch, tmp_path: Path) -> None:
    module = _load_score_module()
    config = load_config(CONFIG_PATH)
    record = _record("Draw a blue square.")
    pdf_path = tmp_path / "candidate.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")
    image_path = tmp_path / "candidate.png"
    image_path.write_text("png", encoding="utf-8")

    class _Scorer:
        backend_name = "fake"
        model_name = "fake-model"

        def score_pairs(self, image_paths, descriptions, *, batch_size):
            assert image_paths == [image_path]
            assert descriptions == ["Draw a blue square."]
            assert batch_size == 4
            return [0.71]

    monkeypatch.setattr(module, "rasterize_pdf", lambda pdf, out, render_config=None: image_path)
    monkeypatch.setattr(module, "prepare_image_for_reward_encoder", lambda image, render_config: image)

    compiled = [module.CompileResult(index=0, record=record, compile_ok=True, pdf_path=pdf_path)]
    module._apply_alignment_scores(
        compiled,
        scorer=_Scorer(),
        config=config,
        output_root=tmp_path,
        batch_size=4,
        on_alignment_error="fail",
    )

    metadata = compiled[0].record["metadata"]
    assert metadata["alignment_score"] == 0.71
    assert metadata["sample_weight"] == 0.71


def test_compile_record_skips_truncated_records(tmp_path: Path) -> None:
    module = _load_score_module()
    record = _record("Draw a line.")
    record["metadata"] = {"is_truncated": True}

    class _Compiler:
        def compile_document(self, *args, **kwargs):
            raise AssertionError("truncated records should not compile")

    result = module._compile_record(0, record, _Compiler(), tmp_path)

    assert result.compile_ok is False
    assert result.pdf_path is None
    assert result.record["metadata"]["compile_ok"] is False
    assert result.record["metadata"]["sample_weight"] == 0.0
