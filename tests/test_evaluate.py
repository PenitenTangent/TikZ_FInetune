from tikz_mlx.evaluate import (
    build_evaluation_record,
    compilation_rate,
    summarize_evaluations,
    tex_edit_distance,
    tokenize_tikz,
)
from tikz_mlx.schemas import CompileStatus


def test_tokenize_tikz_captures_commands_and_numbers() -> None:
    tokens = tokenize_tikz(r"\draw (0,0) -- (1.5,2);")
    assert r"\draw" in tokens
    assert "1.5" in tokens


def test_tex_edit_distance_zero_for_identical_inputs() -> None:
    code = r"\draw (0,0) -- (1,1);"
    assert tex_edit_distance(code, code) == 0


def test_compilation_rate_counts_successes() -> None:
    rate = compilation_rate(
        [
            CompileStatus.SUCCESS,
            CompileStatus.FATAL_ERROR,
            CompileStatus.SUCCESS,
        ]
    )
    assert rate == 2 / 3


def test_summarize_evaluations_reports_attempt_efficiency() -> None:
    records = [
        build_evaluation_record(compiled=True, code=r"\draw (0,0) -- (1,1);", attempt_count=1),
        build_evaluation_record(compiled=False, code=r"\draw (0,0) -- (1,0);", attempt_count=3),
    ]
    summary = summarize_evaluations(records)
    assert summary["average_attempts"] == 2.0
    assert summary["average_attempts_to_success"] == 1.0
