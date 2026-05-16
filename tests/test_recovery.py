import json
from pathlib import Path

import pytest

from tikz_mlx.prompting import build_generation_prompt
from tikz_mlx.recovery import (
    build_eval_manifest,
    evaluate_ab_result_gate,
    filter_quality_records,
    has_repetition_failure,
    quality_filter_record,
    select_equal_mode_sample_ids,
    select_mode_capped_records,
    select_stability_checkpoint,
    stability_checkpoint_from_dict,
    synthetic_repetition_examples,
    validate_contract_file,
    validate_split_disjoint,
    validate_sweep_resume_adapter,
)


def _record(sample_id: str, mode: str) -> dict:
    return {
        "sample_id": sample_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_generation_prompt("Draw."),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "\\begin{tikzpicture}\\end{tikzpicture}\n```\n"}],
            },
        ],
        "metadata": {"generation_mode": mode},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_eval_manifest_sets_are_disjoint_and_equal_mode_stratified(tmp_path: Path) -> None:
    modes = ["plain_tikz", "pgfplots_axis", "graph_nodes", "commutative_diagram", "scientific_schematic"]
    records = [_record(f"{mode}-{idx}", mode) for mode in modes for idx in range(80)]
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset, records)

    manifest = build_eval_manifest(dataset, seed=17)

    all_ids = []
    for values in manifest["sets"].values():
        all_ids.extend(values)
    assert len(all_ids) == len(set(all_ids))
    assert manifest["mode_counts"]["ablation_100"] == {mode: 20 for mode in sorted(modes)}
    assert manifest["mode_counts"]["promotion_120"] == {mode: 24 for mode in sorted(modes)}


def test_contract_validation_catches_assistant_document_wrapper(tmp_path: Path) -> None:
    bad = _record("bad", "plain_tikz")
    bad["messages"][1]["content"][0]["text"] = "\\documentclass{article}\n```\n"
    dataset = tmp_path / "bad.jsonl"
    _write_jsonl(dataset, [bad])

    result = validate_contract_file(dataset)

    assert result["failure_count"] == 1
    assert "assistant_documentclass" in result["failures"][0]["violations"]


def test_validate_split_disjoint_rejects_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(train, [_record("same", "plain_tikz")])
    _write_jsonl(val, [_record("same", "plain_tikz")])
    _write_jsonl(gold, [_record("other", "plain_tikz")])

    with pytest.raises(RuntimeError, match="overlap"):
        validate_split_disjoint({"train": train, "val": val, "gold": gold})


def test_select_stability_checkpoint_requires_5000_and_prefers_low_emd_then_latest() -> None:
    checkpoints = [
        stability_checkpoint_from_dict(
            {"checkpoint_path": "1000.safetensors", "iteration": 1000, "partial_emd": 0.20, "compile_rate": 0.9}
        ),
        stability_checkpoint_from_dict(
            {"checkpoint_path": "5000.safetensors", "iteration": 5000, "partial_emd": 0.20, "compile_rate": 0.9}
        ),
    ]

    selected = select_stability_checkpoint(checkpoints, base_stability_emd=0.25)

    assert selected.iteration == 5000


def test_validate_sweep_resume_rejects_missing_archive(tmp_path: Path) -> None:
    metadata = tmp_path / "adapter.metadata.json"
    metadata.write_text(json.dumps({"promoted": True, "adapter_sha256": "abc"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="archive manifest"):
        validate_sweep_resume_adapter(metadata, tmp_path / "missing.json")


def test_quality_filter_rejects_stealth_and_repeated_line_loops() -> None:
    stealth = _record("stealth", "plain_tikz")
    stealth["messages"][1]["content"][0]["text"] = (
        "\\begin{tikzpicture}[\n"
        + "\n".join(["  >=stealth,"] * 10)
        + "\n]\n\\draw (0,0) -- (1,1);\n\\end{tikzpicture}\n```\n"
    )
    repeated = _record("repeated", "plain_tikz")
    repeated["messages"][1]["content"][0]["text"] = (
        "\\begin{tikzpicture}\n"
        + "\n".join([r"\draw (0,0) -- (1,1);"] * 5)
        + "\n\\end{tikzpicture}\n```\n"
    )

    stealth_reasons, _ = quality_filter_record(stealth)
    repeated_reasons, _ = quality_filter_record(repeated)

    assert "stealth_loop" in stealth_reasons
    assert "repeated_line_loop" in repeated_reasons


def test_filter_quality_records_keeps_clean_and_audits_rejections() -> None:
    clean = _record("clean", "plain_tikz")
    bad = _record("bad", "plain_tikz")
    bad["messages"][1]["content"][0]["text"] = "\\begin{tikzpicture}\n" + "\n".join(["  >=stealth,"] * 10) + "\n\\end{tikzpicture}\n```\n"

    kept, audit = filter_quality_records([clean, bad])

    assert [record["sample_id"] for record in kept] == ["clean"]
    assert audit["total_rejected"] == 1
    assert audit["reason_counts"]["stealth_loop"] == 1
    assert audit["quality_filter_config_hash"]


def test_mode_capped_records_respects_caps_and_keeps_uncapped_modes() -> None:
    records = [_record(f"plain-{idx}", "plain_tikz") for idx in range(5)]
    records.extend(_record(f"graph-{idx}", "graph_nodes") for idx in range(3))

    selected, audit = select_mode_capped_records(
        records,
        caps={"plain_tikz": 2, "graph_nodes": None},
        seed=1,
    )

    modes = [record["metadata"]["generation_mode"] for record in selected]
    assert modes.count("plain_tikz") == 2
    assert modes.count("graph_nodes") == 3
    assert audit["output_mode_counts"] == {"graph_nodes": 3, "plain_tikz": 2}


def test_equal_mode_sample_selection_can_reserve_rare_mode_training_rows() -> None:
    records = [_record(f"plain-{idx}", "plain_tikz") for idx in range(100)]
    records.extend(_record(f"rare-{idx}", "commutative_diagram") for idx in range(12))

    selected = select_equal_mode_sample_ids(
        records,
        total=20,
        seed=1,
        min_remaining_per_mode=10,
    )

    assert sum(sample_id.startswith("rare-") for sample_id in selected) == 2
    assert len(selected) == 20


def test_synthetic_repetition_sidecar_examples_trigger_detector() -> None:
    examples = synthetic_repetition_examples()

    assert examples
    assert all(has_repetition_failure(str(example["text"])) for example in examples)


def test_ablation_gate_rejects_repetition_loop_rate() -> None:
    result = evaluate_ab_result_gate(
        {
            "base": {"compile_rate": 0.50},
            "iter_500": {
                "compile_rate": 0.60,
                "repetition_loop_rate": 0.03,
                "truncation_rate": 0.0,
            },
        },
        candidate_key="iter_500",
        gate="ablation",
        gate_config={
            "promotion_min_compile_rate": 0.20,
            "repetition_loop_rate_max": 0.02,
            "truncation_rate_max": 0.10,
        },
    )

    assert result["passed"] is False
    assert result["checks"]["repetition_loop_rate"]["passed"] is False
