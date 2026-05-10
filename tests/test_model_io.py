from dataclasses import dataclass

from tikz_mlx.model_io import MlxVlmAdapter


@dataclass
class _TextResult:
    text: str


def test_coerce_generation_text_prefers_raw_string() -> None:
    assert MlxVlmAdapter._coerce_generation_text("hello") == "hello"


def test_coerce_generation_text_supports_text_attribute() -> None:
    assert MlxVlmAdapter._coerce_generation_text(_TextResult(text="tikz")) == "tikz"


def test_coerce_generation_text_supports_openai_like_choices() -> None:
    payload = {"choices": [{"message": {"content": "\\begin{tikzpicture}..."}}]}
    assert MlxVlmAdapter._coerce_generation_text(payload) == "\\begin{tikzpicture}..."


def test_coerce_generation_text_supports_string_lists() -> None:
    assert MlxVlmAdapter._coerce_generation_text(["first", "second"]) == "first"


def test_supports_parameter_detects_optional_kwargs() -> None:
    def fn(a, *, min_p=None):
        return a

    assert MlxVlmAdapter._supports_parameter(fn, "min_p") is True
    assert MlxVlmAdapter._supports_parameter(fn, "repetition_penalty") is False
