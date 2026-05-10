from tikz_mlx.dataset import (
    detect_generation_mode,
    extract_coordinate_bounding_box,
    extract_tikz_libraries,
    sample_to_stage2_record,
    sample_to_training_record,
)
from tikz_mlx.schemas import TikzSample


def _sample(code: str, environment: str = "tikzpicture") -> TikzSample:
    return TikzSample(
        sample_id="sample-1",
        source="unit",
        raw_code=code,
        normalized_code=code,
        environment=environment,
        description="Draw the requested figure.",
        metadata={"dataset_id": "unit"},
    )


def test_extracts_libraries_and_bounding_box() -> None:
    code = (
        "\\usetikzlibrary{calc, arrows.meta}\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture}\\draw (-0.2,-0.2) -- (4.25,3);\\end{tikzpicture}\n"
        "\\end{document}"
    )

    assert extract_tikz_libraries(code) == ["arrows.meta", "calc"]
    assert extract_coordinate_bounding_box(code) == {
        "min_x": -0.2,
        "min_y": -0.2,
        "max_x": 4.25,
        "max_y": 3.0,
    }


def test_detect_generation_mode_from_tex() -> None:
    assert detect_generation_mode("\\begin{axis}\\addplot coordinates {(0,0)};\\end{axis}") == "pgfplots_axis"
    assert detect_generation_mode("\\begin{tikzcd} A \\arrow[r] & B \\end{tikzcd}") == "commutative_diagram"
    assert detect_generation_mode("\\graph { a -- b };") == "graph_nodes"
    assert detect_generation_mode("\\begin{tikzpicture}\\draw (0,0)--(1,1);\\end{tikzpicture}") == "plain_tikz"


def test_training_record_includes_mode_and_geometry_hints_in_metadata_and_prompt() -> None:
    record = sample_to_training_record(
        _sample("\\documentclass{article}\n\\usepackage{tikz}\n\\begin{document}\\begin{tikzpicture}\\draw (0,0) -- (2,1);\\end{tikzpicture}\\end{document}")
    )

    metadata = record["metadata"]
    prompt = record["messages"][0]["content"][0]["text"]
    assistant = record["messages"][1]["content"][0]["text"]
    assert metadata["generation_mode"] == "plain_tikz"
    assert metadata["geometry_hints"]["bounding_box"]["max_x"] == 2.0
    assert "[GEOMETRY HINTS]" in prompt
    assert "mode: plain_tikz" in prompt
    assert "bounding_box: (0, 0) to (2, 1)" in prompt
    assert prompt.count("--- Starting Preamble ---") == 1
    assert prompt.count("\\documentclass") == 1
    assert prompt.rstrip().endswith("```latex")
    assert assistant.count("```") == 1
    assert "\\begin{document}" not in assistant


def test_stage2_record_uses_same_mode_hints() -> None:
    record = sample_to_stage2_record(_sample("\\begin{axis}\\addplot coordinates {(0,0) (1,1)};\\end{axis}"))

    assert record["metadata"]["generation_mode"] == "pgfplots_axis"
    assert "mode: pgfplots_axis" in record["prompt_text"]
