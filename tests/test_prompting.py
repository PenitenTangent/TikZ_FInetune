from tikz_mlx.prompting import (
    build_compile_repair_prompt,
    build_generation_prompt,
    build_visual_repair_prompt,
)
from tikz_mlx.schemas import CompileStatus, CompileSummary


def _compile_summary() -> CompileSummary:
    return CompileSummary(
        status=CompileStatus.RECOVERABLE_ERROR,
        return_code=1,
        key_errors=["Undefined control sequence"],
        line_hints=[12],
        missing_packages=["circuitikz"],
        stdout="",
        stderr="",
        log_text="",
        elapsed_seconds=0.1,
    )


def test_build_generation_prompt_includes_contract_and_output_constraints() -> None:
    prompt = build_generation_prompt(" Draw a right triangle with vertices A, B, and C. ")

    assert "Generate only the TikZ environment body" in prompt
    assert "Draw a right triangle with vertices A, B, and C." in prompt
    assert "Do not output a LaTeX preamble" in prompt
    assert "\\documentclass[tikz]{standalone}" not in prompt
    assert prompt.rstrip().endswith("```latex")


def test_build_compile_repair_prompt_emphasizes_intent_and_minimal_edits() -> None:
    prompt = build_compile_repair_prompt("\\begin{document}\\end{document}", _compile_summary())

    assert "preserving the figure intent" in prompt
    assert "Keep edits minimal" in prompt
    assert "no Markdown fences" in prompt
    assert "Undefined control sequence" in prompt
    assert "circuitikz" in prompt


def test_build_visual_repair_prompt_forbids_debug_artifacts_and_markdown() -> None:
    prompt = build_visual_repair_prompt(
        "Draw two concentric circles.",
        "\\begin{tikzpicture}\\draw (0,0) circle (1);\\end{tikzpicture}",
    )

    assert "Do not add grid code" in prompt
    assert "Do not include Markdown fences" in prompt
    assert "Draw two concentric circles." in prompt
