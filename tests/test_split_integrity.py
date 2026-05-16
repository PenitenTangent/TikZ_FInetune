import json

from tools.validate_split_integrity import (
    validate_ngram_contamination,
    validate_no_cross_split_prompt_hash_overlap,
    validate_no_cross_split_target_hash_overlap,
    validate_source_uniqueness,
)


def _record(sample_id: str, prompt: str, code: str, source: str | None = None) -> dict:
    record = {
        "sample_id": sample_id,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": code}]},
        ],
    }
    if source is not None:
        record["metadata"] = {"source_id": source}
    return record


def test_split_integrity_detects_exact_target_overlap() -> None:
    train = [_record("train-a", "draw a line", r"\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}")]
    val = [_record("val-a", "different prompt", r"\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}")]

    violations = validate_no_cross_split_target_hash_overlap({"train": train, "val": val})

    assert [v.type for v in violations] == ["cross_split_target_hash_overlap"]


def test_split_integrity_detects_exact_prompt_overlap() -> None:
    train = [_record("train-a", "draw a diagonal line", r"\draw (0,0)--(1,1);")]
    val = [_record("val-a", "draw a diagonal line", r"\draw (0,0)--(2,2);")]

    violations = validate_no_cross_split_prompt_hash_overlap({"train": train, "val": val})

    assert [v.type for v in violations] == ["cross_split_prompt_hash_overlap"]


def test_split_integrity_detects_ngram_code_contamination() -> None:
    code = r"\draw (0,0) -- (1,1); \node at (0,0) {A}; \fill (1,1) circle (2pt);"
    train = [_record("train-a", "train prompt", code)]
    val = [_record("val-a", "held out prompt", code + r" \draw (2,2)--(3,3);")]

    violations = validate_ngram_contamination(
        {"train": train, "val": val},
        code_shingle_size=4,
        prompt_shingle_size=4,
        max_code_containment=0.20,
        max_prompt_containment=0.50,
    )

    assert any(v.type == "code_ngram_contamination" for v in violations)


def test_split_integrity_detects_source_reuse_across_eval_boundary() -> None:
    train = [_record("train-a", "train prompt", r"\draw (0,0)--(1,1);", source="paper-1")]
    gold = [_record("gold-a", "held out prompt", r"\draw (2,2)--(3,3);", source="paper-1")]

    violations = validate_source_uniqueness({"train": train, "gold_eval": gold})

    assert [v.type for v in violations] == ["source_reused_across_eval_boundary"]
