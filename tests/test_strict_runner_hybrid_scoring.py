from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_strict_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "stage1_ab_eval_strict_runner.py"
    if not module_path.exists():
        module_path = Path(__file__).resolve().parents[1] / "outputs" / "stage1_ab_eval_strict_runner.py"
    spec = importlib.util.spec_from_file_location("stage1_ab_eval_strict_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load strict runner module for testing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_a_weighted_score_prefers_token_jaccard_but_keeps_structure_signal() -> None:
    module = _load_strict_runner_module()

    ref = r"""\documentclass[tikz]{standalone}
\begin{document}
\begin{tikzpicture}
  \draw (0,0) -- (2,0) -- (1,1.5) -- cycle;
\end{tikzpicture}
\end{document}
"""

    same = ref
    commented = r"""\documentclass[tikz]{standalone}
\begin{document}
\begin{tikzpicture}
  % commentary only
  \draw (0,0) -- (2,0) -- (1,1.5) -- cycle; % same geometry
\end{tikzpicture}
\end{document}
"""

    different = r"""\documentclass[tikz]{standalone}
\begin{document}
\begin{tikzpicture}
  \fill (0,0) rectangle (2,2);
\end{tikzpicture}
\end{document}
"""

    token_weight = 0.75
    command_weight = 0.25

    def phase_a(code: str) -> float:
        tj = module._token_jaccard(ref, code)
        cm = module._command_number_mix(ref, code)
        return (token_weight * tj + command_weight * cm) / (token_weight + command_weight)

    s_same = phase_a(same)
    s_comment = phase_a(commented)
    s_diff = phase_a(different)

    assert s_same > 0.99
    assert s_comment > 0.85
    assert s_diff < 0.80
    assert s_same >= s_comment >= s_diff


class _AlwaysFailEMD:
    def score_embeddings(self, reference, candidate):
        raise RuntimeError("emd failed")


class _ConstantSelfSim:
    def __init__(self, value: float):
        self.value = value

    def score_embeddings(self, reference, candidate):
        return self.value


class _ConstantEMD:
    def __init__(self, value: float):
        self.value = value

    def score_embeddings(self, reference, candidate):
        return self.value


def test_phase_b_uses_selfsim_only_when_emd_fails() -> None:
    module = _load_strict_runner_module()
    ref = [[1.0, 0.0], [0.0, 1.0]]
    cand = [[1.0, 0.0], [0.0, 1.0]]

    fallback = module._score_with_fallback(ref, cand, _AlwaysFailEMD(), _ConstantSelfSim(0.61))
    assert fallback["backend_used"] == "selfsim"
    assert fallback["fallback_used"] is True
    assert fallback["score"] == 0.61
    assert "emd failed" in fallback["fallback_reason"]

    primary = module._score_with_fallback(ref, cand, _ConstantEMD(0.98), _ConstantSelfSim(0.11))
    assert primary["backend_used"] == "emd"
    assert primary["fallback_used"] is False
    assert primary["score"] == 0.98
    assert primary["fallback_reason"] is None


def test_hybrid_score_uses_phase_b_only_for_promotion(monkeypatch, tmp_path: Path) -> None:
    module = _load_strict_runner_module()

    class _DummyEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_image(self, image_path: str):
            return [[1.0, 0.0], [0.0, 1.0]]

        def unload(self) -> None:
            return None

    monkeypatch.setattr(module, "FrozenDetikzifyEncoder", _DummyEncoder)
    monkeypatch.setattr(
        module,
        "rasterize_pdf",
        lambda pdf_path, output_dir, render_config=None: tmp_path / "candidate.png",
    )
    monkeypatch.setattr(module, "prepare_image_for_reward_encoder", lambda image_path, render_config: image_path)

    cfg = type(
        "Cfg",
        (),
        {
            "memory": object(),
            "paths": type("Paths", (), {"root_dir": tmp_path})(),
        },
    )()
    scorer = module.HybridScorer(
        cfg=cfg,
        compiler=None,
        out_root=tmp_path,
        similarity_mode="hybrid",
        prefilter_threshold=0.80,
        visual_score_threshold=0.75,
        reward_backend="emd",
        reward_model_id="dummy",
        reference_code_by_row={0: r"\draw (0,0) -- (1,1);"},
        render_config=object(),
        phase_a_token_weight=0.75,
        phase_a_command_weight=0.25,
        phase_b_blend_weight=0.70,
        hybrid_combine_mode="multiplicative",
        hybrid_score_gamma=1.4,
    )
    monkeypatch.setattr(scorer, "_get_reference_embedding", lambda row_index: [[1.0, 0.0], [0.0, 1.0]])

    result = scorer.score(
        sample={"row_index": 0},
        normalized_code=r"\fill (0,0) rectangle (2,2);",
        substantive_compile_success=True,
        sample_dir=tmp_path,
        candidate_pdf_path=str(tmp_path / "candidate.pdf"),
    )

    assert result["phase_a_score"] < 0.80
    assert result["code_prefilter_pass"] is False
    assert result["phase_b_score"] == pytest.approx(1.0)
    assert result["hybrid_score"] == pytest.approx(1.0)
    assert result["hybrid_pass"] is True
    assert result["hybrid_pass_reason"] == "phase_b_only"
