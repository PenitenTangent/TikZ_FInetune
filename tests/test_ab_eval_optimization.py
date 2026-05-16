import json
from dataclasses import dataclass
from pathlib import Path

from tools.ab_eval import _base_cache_key, _score_generated_sample, _select_manifest_samples, _variant_metrics
from tools.check_promotion_gate import evaluate_promotion_gate


def test_manifest_selection_respects_quick_gate_limit(tmp_path: Path) -> None:
    pool = [{"sample_id": f"s{i}", "prompt_text": "p", "reference_code": "r"} for i in range(5)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sample_ids": ["s3", "s1", "s4", "s0"]}), encoding="utf-8")

    selected = _select_manifest_samples(pool, manifest, limit=2)

    assert [sample["sample_id"] for sample in selected] == ["s3", "s1"]


@dataclass
class _Model:
    model_id: str = "unit-model"
    enable_thinking: bool = False


@dataclass
class _Compiler:
    tectonic_binary: str
    timeout_seconds: int = 15


@dataclass
class _Cfg:
    model: _Model
    compiler: _Compiler


def test_base_cache_key_changes_with_compiler_config(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    samples = [{"sample_id": "s0"}]
    common = {
        "dataset_path": dataset,
        "manifest_path": None,
        "samples": samples,
        "seed": 42,
        "max_tokens": 128,
        "prompt_contract_version": "unit",
        "prompt_template_sha256_value": "abc",
    }

    key_a = _base_cache_key(cfg=_Cfg(_Model(), _Compiler("tectonic-a")), **common)
    key_b = _base_cache_key(cfg=_Cfg(_Model(), _Compiler("tectonic-b")), **common)

    assert key_a != key_b


def test_score_generated_sample_treats_empty_output_as_failed_quality() -> None:
    result = _score_generated_sample(
        {
            "sample": {"sample_id": "s0", "prompt_text": "p", "reference_code": "r"},
            "raw_response": "",
            "max_tokens": 128,
        },
        compiler_config=_Compiler("tectonic"),
    )

    assert result["true_substantive"] is False
    assert result["bad_patterns_pass"] is False
    assert result["bad_pattern_violations"] == ["empty_output"]


def _result(sample_id: str, raw_tokens: int, code_tokens: int) -> dict:
    return {
        "sample_id": sample_id,
        "compile_ok": True,
        "true_substantive": True,
        "bad_patterns_pass": True,
        "has_preview_env": False,
        "has_usepackage": False,
        "has_documentclass": False,
        "has_decorations_geometric": False,
        "repetition_loop": False,
        "closing_fence_exactly_once": True,
        "truncated": False,
        "code_length": code_tokens,
        "raw_token_length": raw_tokens,
        "code_token_length": code_tokens,
    }


def test_variant_metrics_include_average_token_lengths() -> None:
    base = [_result("b0", 100, 50)]
    candidate = [_result("c0", 150, 75)]

    metrics = _variant_metrics("finetuned", candidate, base)

    assert metrics["avg_raw_token_length"] == 150
    assert metrics["avg_code_token_length"] == 75
    assert metrics["avg_raw_token_ratio_vs_base"] == 1.5
    assert metrics["avg_code_token_ratio_vs_base"] == 1.5


def test_promotion_gate_rejects_excessive_average_token_ratio() -> None:
    passing_common = {
        "compile_rate": 1.0,
        "substantive_rate": 1.0,
        "bad_pattern_pass_rate": 1.0,
        "preview_environment_rate": 0.0,
        "assistant_usepackage_rate": 0.0,
        "assistant_documentclass_rate": 0.0,
        "decorations_geometric_rate": 0.0,
        "repetition_loop_rate": 0.0,
        "closing_fence_exactly_once_rate": 1.0,
        "avg_code_length_ratio_vs_base": 1.0,
    }
    result = evaluate_promotion_gate(
        {
            "base": {**passing_common, "avg_raw_token_ratio_vs_base": 1.0, "avg_code_token_ratio_vs_base": 1.0},
            "finetuned": {
                **passing_common,
                "avg_raw_token_ratio_vs_base": 2.0,
                "avg_code_token_ratio_vs_base": 1.2,
            },
        }
    )

    assert result["pass"] is False
    assert any("avg_raw_token_ratio_vs_base" in violation for violation in result["violations"])
