from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tikz_mlx.prompting import PROMPT_CONTRACT_VERSION, prompt_template_sha256
from tikz_mlx.dataset import validate_row_aligned_example_indices


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools" / "build_stage0_compile_curriculum.py"
spec = importlib.util.spec_from_file_location("build_stage0_compile_curriculum", BUILDER_PATH)
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


class _AlwaysPassCompiler:
    pass


def _assistant_text(record: dict) -> str:
    for message in record["messages"]:
        if message["role"] == "assistant":
            return message["content"]
    raise AssertionError("missing assistant message")


def test_stage0_compile_curriculum_writer_emits_contract_rows(tmp_path: Path) -> None:
    output = tmp_path / "stage0.jsonl"
    audit_output = tmp_path / "stage0_audit.json"
    candidates = builder.build_candidates(
        primitive_count=4,
        repair_count=4,
        mixed_count=2,
        outputs_dir=tmp_path / "outputs",
    )

    audit = builder.write_dataset(
        candidates=candidates,
        output=output,
        audit_output=audit_output,
        compiler=_AlwaysPassCompiler(),
        requested_count=10,
        outputs_dir=tmp_path / "outputs",
        compile_gate=False,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10
    validate_row_aligned_example_indices([row["example_index"] for row in rows])
    assert audit["emitted_row_count"] == 10
    assert audit["requested_row_count"] == 10
    assert audit["category_counts"]["primitive_line"] == 4
    assert audit["prompt_template_sha256"] == prompt_template_sha256()
    assert audit_output.exists()

    for index, row in enumerate(rows):
        assert row["example_index"] == index
        assert row["metadata"]["example_index"] == index
        assert row["metadata"]["prompt_contract_version"] == PROMPT_CONTRACT_VERSION
        assert row["metadata"]["target_contract"] == "body_only_environment"
        assert row["metadata"]["generation_mode"] == "plain_tikz"
        assert row["messages"][0]["content"].rstrip().endswith("```latex")
        assistant = _assistant_text(row)
        assert assistant.count("```") == 1
        assert assistant.rstrip().endswith("```")
        assert "\\begin{document}" not in assistant
        assert "\\documentclass" not in assistant


def test_stage0_repair_patterns_avoid_raw_invalid_forms() -> None:
    repairs = list(builder.iter_repair_candidates(80, outputs_dir=Path("missing_outputs")))
    assert repairs

    invalid_tokens = [
        "light blue",
        "dark gray",
        "transparent",
        "some_other_node",
        "(\\hat",
        " curve ",
        "Placeholder",
    ]
    for candidate in repairs:
        assert candidate.category == "repair_pattern"
        assert candidate.code.count("\\begin{tikzpicture}") == 1
        assert candidate.code.count("\\end{tikzpicture}") == 1
        assert "```" not in candidate.code
        assert "%" not in candidate.code
        for token in invalid_tokens:
            assert token not in candidate.code
