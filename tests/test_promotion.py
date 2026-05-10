import json
from pathlib import Path

import pytest

from tikz_mlx.promotion import is_canonical_tikz_document, run_sft_promotion_gate


CANONICAL_DOC = r"""\documentclass[tikz]{standalone}
\begin{document}
\begin{tikzpicture}
\draw (0,0) -- (1,1);
\end{tikzpicture}
\end{document}"""

FENCED_DOC = """```latex
\\documentclass[tikz]{standalone}
\\begin{document}
\\end{document}
```"""

DUPLICATED_WRAPPER_DOC = r"""\documentclass[tikz]{standalone}
\begin{document}
\documentclass[tikz]{standalone}
\begin{document}
\end{document}
\end{document}"""


def _write_report(
    path: Path,
    *,
    compile_rate: float,
    per_prompt: list[dict[str, object]],
    schema_rate: float | None = None,
) -> None:
    payload: dict[str, object] = {
        "compile_rate": compile_rate,
        "per_prompt": per_prompt,
    }
    if schema_rate is not None:
        payload["schema_rate"] = schema_rate
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_strict_report(path: Path, *, base: dict[str, object], stage1: dict[str, object]) -> None:
    payload = {
        "sample_size": 20,
        "base": base,
        "stage1": stage1,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_is_canonical_tikz_document_accepts_normalized_document() -> None:
    assert is_canonical_tikz_document(CANONICAL_DOC)


def test_is_canonical_tikz_document_rejects_markdown_fences() -> None:
    assert not is_canonical_tikz_document(FENCED_DOC)


def test_is_canonical_tikz_document_rejects_duplicated_wrappers() -> None:
    assert not is_canonical_tikz_document(DUPLICATED_WRAPPER_DOC)


def test_run_sft_promotion_gate_promotes_when_thresholds_pass(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline.json"
    candidate_report = tmp_path / "candidate.json"

    _write_report(
        baseline_report,
        compile_rate=0.30,
        per_prompt=[{"code": CANONICAL_DOC}, {"code": FENCED_DOC}],
    )
    _write_report(
        candidate_report,
        compile_rate=0.55,
        per_prompt=[{"code": CANONICAL_DOC}, {"code": CANONICAL_DOC}],
    )

    candidate_checkpoint = tmp_path / "candidate_adapters.safetensors"
    candidate_checkpoint.write_bytes(b"candidate")

    sft_final_path = tmp_path / "sft_final.safetensors"
    policy_init_path = tmp_path / "policy_init.safetensors"

    result = run_sft_promotion_gate(
        baseline_report_path=baseline_report,
        candidate_report_path=candidate_report,
        baseline_key=None,
        candidate_key=None,
        min_compile_delta=0.10,
        min_schema_delta=0.10,
        min_candidate_compile_rate=0.50,
        min_candidate_schema_rate=0.70,
        baseline_compile_rate=None,
        candidate_compile_rate=None,
        promote=True,
        candidate_checkpoint_path=candidate_checkpoint,
        sft_final_path=sft_final_path,
        policy_init_path=policy_init_path,
        force_policy_init=False,
        run_id="promotion-test",
    )

    assert result["gate"]["passed"] is True
    assert result["promotion"] is not None
    assert sft_final_path.read_bytes() == b"candidate"
    assert policy_init_path.read_bytes() == b"candidate"
    assert (tmp_path / "sft_final.safetensors.metadata.json").exists()


def test_run_sft_promotion_gate_does_not_promote_on_gate_failure(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline.json"
    candidate_report = tmp_path / "candidate.json"

    _write_report(
        baseline_report,
        compile_rate=0.60,
        per_prompt=[{"code": CANONICAL_DOC}, {"code": CANONICAL_DOC}],
    )
    _write_report(
        candidate_report,
        compile_rate=0.50,
        per_prompt=[{"code": CANONICAL_DOC}, {"code": FENCED_DOC}],
    )

    candidate_checkpoint = tmp_path / "candidate_adapters.safetensors"
    candidate_checkpoint.write_bytes(b"candidate")

    result = run_sft_promotion_gate(
        baseline_report_path=baseline_report,
        candidate_report_path=candidate_report,
        baseline_key=None,
        candidate_key=None,
        min_compile_delta=0.0,
        min_schema_delta=0.0,
        min_candidate_compile_rate=0.0,
        min_candidate_schema_rate=0.0,
        baseline_compile_rate=None,
        candidate_compile_rate=None,
        promote=True,
        candidate_checkpoint_path=candidate_checkpoint,
        sft_final_path=tmp_path / "sft_final.safetensors",
        policy_init_path=tmp_path / "policy_init.safetensors",
        force_policy_init=False,
        run_id="promotion-test",
    )

    assert result["gate"]["passed"] is False
    assert result["promotion"] is None


def test_run_sft_promotion_gate_uses_recovery_gate_config(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline.json"
    candidate_report = tmp_path / "candidate.json"
    gate_config = tmp_path / "gate_config.json"
    _write_report(
        baseline_report,
        compile_rate=0.30,
        per_prompt=[{"code": CANONICAL_DOC}],
        schema_rate=0.80,
    )
    _write_report(
        candidate_report,
        compile_rate=0.19,
        per_prompt=[{"code": CANONICAL_DOC}],
        schema_rate=0.90,
    )
    gate_config.write_text(json.dumps({"promotion_min_compile_rate": 0.20}), encoding="utf-8")

    result = run_sft_promotion_gate(
        baseline_report_path=baseline_report,
        candidate_report_path=candidate_report,
        baseline_key=None,
        candidate_key=None,
        min_compile_delta=0.0,
        min_schema_delta=0.0,
        min_candidate_compile_rate=0.0,
        min_candidate_schema_rate=0.0,
        baseline_compile_rate=None,
        candidate_compile_rate=None,
        promote=False,
        candidate_checkpoint_path=None,
        sft_final_path=tmp_path / "sft_final.safetensors",
        policy_init_path=tmp_path / "policy_init.safetensors",
        force_policy_init=False,
        run_id="promotion-test",
        gate_config_path=gate_config,
    )

    assert result["gate"]["checks"]["candidate_compile_floor"]["required"] == 0.20
    assert result["gate"]["passed"] is False


def test_run_sft_promotion_gate_rejects_repetition_and_truncation(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline.json"
    candidate_report = tmp_path / "candidate.json"
    gate_config = tmp_path / "gate_config.json"
    _write_report(
        baseline_report,
        compile_rate=0.50,
        per_prompt=[{"code": CANONICAL_DOC}],
        schema_rate=1.0,
    )
    _write_report(
        candidate_report,
        compile_rate=0.60,
        per_prompt=[{"code": CANONICAL_DOC}],
        schema_rate=1.0,
    )
    candidate_payload = json.loads(candidate_report.read_text(encoding="utf-8"))
    candidate_payload["repetition_loop_rate"] = 0.03
    candidate_payload["truncation_rate"] = 0.11
    candidate_report.write_text(json.dumps(candidate_payload), encoding="utf-8")
    gate_config.write_text(
        json.dumps(
            {
                "promotion_min_compile_rate": 0.20,
                "repetition_loop_rate_max": 0.02,
                "truncation_rate_max": 0.10,
            }
        ),
        encoding="utf-8",
    )

    result = run_sft_promotion_gate(
        baseline_report_path=baseline_report,
        candidate_report_path=candidate_report,
        baseline_key=None,
        candidate_key=None,
        min_compile_delta=0.0,
        min_schema_delta=0.0,
        min_candidate_compile_rate=0.0,
        min_candidate_schema_rate=0.0,
        baseline_compile_rate=None,
        candidate_compile_rate=None,
        promote=False,
        candidate_checkpoint_path=None,
        sft_final_path=tmp_path / "sft_final.safetensors",
        policy_init_path=tmp_path / "policy_init.safetensors",
        force_policy_init=False,
        run_id="promotion-test",
        gate_config_path=gate_config,
    )

    assert result["gate"]["checks"]["candidate_repetition_loop_ceiling"]["passed"] is False
    assert result["gate"]["checks"]["candidate_truncation_ceiling"]["passed"] is False
    assert result["gate"]["passed"] is False


def test_run_sft_promotion_gate_recomputes_schema_from_per_prompt_records(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline.json"
    candidate_report = tmp_path / "candidate.json"

    _write_report(
        baseline_report,
        compile_rate=0.50,
        per_prompt=[{"code": CANONICAL_DOC}, {"code": CANONICAL_DOC}],
        schema_rate=0.0,
    )
    _write_report(
        candidate_report,
        compile_rate=0.50,
        per_prompt=[{"code": FENCED_DOC}, {"code": FENCED_DOC}],
        schema_rate=1.0,
    )

    result = run_sft_promotion_gate(
        baseline_report_path=baseline_report,
        candidate_report_path=candidate_report,
        baseline_key=None,
        candidate_key=None,
        min_compile_delta=0.0,
        min_schema_delta=0.0,
        min_candidate_compile_rate=0.0,
        min_candidate_schema_rate=0.0,
        baseline_compile_rate=None,
        candidate_compile_rate=None,
        promote=False,
        candidate_checkpoint_path=None,
        sft_final_path=tmp_path / "sft_final.safetensors",
        policy_init_path=tmp_path / "policy_init.safetensors",
        force_policy_init=False,
        run_id="promotion-test",
    )

    assert result["gate"]["baseline"]["schema_rate"] == 1.0
    assert result["gate"]["candidate"]["schema_rate"] == 0.0
    assert result["gate"]["passed"] is False


def test_run_sft_promotion_gate_requires_code_bearing_per_prompt(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline.json"
    candidate_report = tmp_path / "candidate.json"

    baseline_report.write_text(
        json.dumps({"compile_rate": 0.50, "schema_rate": 1.0}),
        encoding="utf-8",
    )
    candidate_report.write_text(
        json.dumps({"compile_rate": 0.50, "schema_rate": 1.0}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="strict schema adherence rate"):
        run_sft_promotion_gate(
            baseline_report_path=baseline_report,
            candidate_report_path=candidate_report,
            baseline_key=None,
            candidate_key=None,
            min_compile_delta=0.0,
            min_schema_delta=0.0,
            min_candidate_compile_rate=0.0,
            min_candidate_schema_rate=0.0,
            baseline_compile_rate=None,
            candidate_compile_rate=None,
            promote=False,
            candidate_checkpoint_path=None,
            sft_final_path=tmp_path / "sft_final.safetensors",
            policy_init_path=tmp_path / "policy_init.safetensors",
            force_policy_init=False,
            run_id="promotion-test",
        )


def test_run_sft_promotion_gate_resolves_schema_from_tex_path(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline.json"
    candidate_report = tmp_path / "candidate.json"
    baseline_tex = tmp_path / "baseline.tex"
    candidate_tex = tmp_path / "candidate.tex"

    baseline_tex.write_text(CANONICAL_DOC, encoding="utf-8")
    candidate_tex.write_text(DUPLICATED_WRAPPER_DOC, encoding="utf-8")
    _write_report(
        baseline_report,
        compile_rate=0.50,
        per_prompt=[{"tex_path": str(baseline_tex)}],
    )
    _write_report(
        candidate_report,
        compile_rate=0.50,
        per_prompt=[{"tex_path": str(candidate_tex)}],
    )

    result = run_sft_promotion_gate(
        baseline_report_path=baseline_report,
        candidate_report_path=candidate_report,
        baseline_key=None,
        candidate_key=None,
        min_compile_delta=0.0,
        min_schema_delta=0.0,
        min_candidate_compile_rate=0.0,
        min_candidate_schema_rate=0.0,
        baseline_compile_rate=None,
        candidate_compile_rate=None,
        promote=False,
        candidate_checkpoint_path=None,
        sft_final_path=tmp_path / "sft_final.safetensors",
        policy_init_path=tmp_path / "policy_init.safetensors",
        force_policy_init=False,
        run_id="promotion-test",
    )

    assert result["gate"]["baseline"]["schema_rate"] == 1.0
    assert result["gate"]["candidate"]["schema_rate"] == 0.0


def test_run_sft_promotion_gate_supports_strict_report_blocks(tmp_path: Path) -> None:
    strict_report = tmp_path / "strict_report.json"
    _write_strict_report(
        strict_report,
        base={
            "substantive_tikz_rate": 0.70,
            "substantive_compile_success_rate": 0.08,
        },
        stage1={
            "substantive_tikz_rate": 0.82,
            "substantive_compile_success_rate": 0.16,
        },
    )

    result = run_sft_promotion_gate(
        baseline_report_path=strict_report,
        candidate_report_path=strict_report,
        baseline_key="base",
        candidate_key="stage1",
        min_compile_delta=0.05,
        min_schema_delta=0.05,
        min_candidate_compile_rate=0.10,
        min_candidate_schema_rate=0.75,
        baseline_compile_rate=None,
        candidate_compile_rate=None,
        promote=False,
        candidate_checkpoint_path=None,
        sft_final_path=tmp_path / "sft_final.safetensors",
        policy_init_path=tmp_path / "policy_init.safetensors",
        force_policy_init=False,
        run_id="promotion-test",
    )

    assert result["gate"]["passed"] is True
    assert result["gate"]["baseline"]["compile_rate"] == pytest.approx(0.08)
    assert result["gate"]["candidate"]["compile_rate"] == pytest.approx(0.16)
    assert result["gate"]["baseline"]["schema_rate"] == pytest.approx(0.70)
    assert result["gate"]["candidate"]["schema_rate"] == pytest.approx(0.82)


def test_run_sft_promotion_gate_prefers_strict_schema_rate_from_hybrid_artifact(tmp_path: Path) -> None:
    strict_report = tmp_path / "strict_hybrid_report.json"
    _write_strict_report(
        strict_report,
        base={
            "schema_rate": 0.18,
            "schema_rate_source": "hybrid_pass_rate",
            "substantive_tikz_rate": 0.70,
            "substantive_compile_success_rate": 0.08,
        },
        stage1={
            "schema_rate": 0.31,
            "schema_rate_source": "hybrid_pass_rate",
            "substantive_tikz_rate": 0.10,
            "substantive_compile_success_rate": 0.12,
        },
    )

    result = run_sft_promotion_gate(
        baseline_report_path=strict_report,
        candidate_report_path=strict_report,
        baseline_key="base",
        candidate_key="stage1",
        min_compile_delta=0.01,
        min_schema_delta=0.10,
        min_candidate_compile_rate=0.10,
        min_candidate_schema_rate=0.30,
        baseline_compile_rate=None,
        candidate_compile_rate=None,
        promote=False,
        candidate_checkpoint_path=None,
        sft_final_path=tmp_path / "sft_final.safetensors",
        policy_init_path=tmp_path / "policy_init.safetensors",
        force_policy_init=False,
        run_id="promotion-test",
    )

    assert result["gate"]["baseline"]["schema_rate"] == pytest.approx(0.18)
    assert result["gate"]["candidate"]["schema_rate"] == pytest.approx(0.31)
    assert result["gate"]["checks"]["schema_delta"]["observed"] == pytest.approx(0.13)
    assert result["gate"]["passed"] is True
