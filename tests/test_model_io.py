from dataclasses import dataclass

from tikz_mlx.model_io import LoadedModel, MlxVlmAdapter
from tikz_mlx.schemas import GenerationRequest


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


def test_generate_passes_no_repeat_ngram_size_when_supported(monkeypatch) -> None:
    calls = {}
    adapter = MlxVlmAdapter.__new__(MlxVlmAdapter)
    adapter.model_config = type(
        "ModelConfig",
        (),
        {"max_output_tokens": 16, "temperature": 0.0, "top_p": 1.0, "top_k": 64, "model_id": "unit"},
    )()

    def stream_generate(
        *,
        model,
        processor,
        prompt,
        image,
        max_tokens,
        temperature,
        top_p,
        top_k,
        verbose,
        no_repeat_ngram_size=None,
    ):
        calls["no_repeat_ngram_size"] = no_repeat_ngram_size
        yield "done"

    monkeypatch.setattr(adapter, "ensure_loaded", lambda: LoadedModel(model=object(), processor=object()))
    monkeypatch.setattr(adapter, "_import_api", lambda: (None, None, stream_generate, None))
    monkeypatch.setattr(adapter, "_format_prompt", lambda request: "prompt")

    result = adapter.generate(GenerationRequest(description="x", no_repeat_ngram_size=4))

    assert "done" in result.text
    assert calls["no_repeat_ngram_size"] == 4


def test_generate_omits_no_repeat_ngram_size_when_unsupported(monkeypatch) -> None:
    calls = {"called": False}
    adapter = MlxVlmAdapter.__new__(MlxVlmAdapter)
    adapter.model_config = type(
        "ModelConfig",
        (),
        {"max_output_tokens": 16, "temperature": 0.0, "top_p": 1.0, "top_k": 64, "model_id": "unit"},
    )()

    def stream_generate(*, model, processor, prompt, image, max_tokens, temperature, top_p, top_k, verbose):
        calls["called"] = True
        yield "done"

    monkeypatch.setattr(adapter, "ensure_loaded", lambda: LoadedModel(model=object(), processor=object()))
    monkeypatch.setattr(adapter, "_import_api", lambda: (None, None, stream_generate, None))
    monkeypatch.setattr(adapter, "_format_prompt", lambda request: "prompt")

    result = adapter.generate(GenerationRequest(description="x", no_repeat_ngram_size=4))

    assert "done" in result.text
    assert calls["called"] is True
