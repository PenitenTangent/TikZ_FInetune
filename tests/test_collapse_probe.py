from dataclasses import dataclass

from tikz_mlx import collapse_probe


@dataclass
class _Result:
    text: str


class _Model:
    def __init__(self) -> None:
        self.mode = "train"

    def eval(self) -> None:
        self.mode = "eval"

    def train(self) -> None:
        self.mode = "train"


def test_collapse_probe_passes_production_decoding_kwargs(monkeypatch) -> None:
    calls = []

    def fake_generate(
        *,
        model,
        processor,
        prompt,
        max_tokens,
        verbose,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
    ):
        calls.append(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
            }
        )
        return _Result("\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n\\end{tikzpicture}")

    monkeypatch.setattr(collapse_probe, "generate", fake_generate)

    passed, failures = collapse_probe.run_collapse_probe(
        _Model(),
        processor=object(),
        build_prompt_fn=lambda text: text,
        decoding={
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 64,
            "min_p": 0.05,
            "repetition_penalty": 1.2,
        },
    )

    assert passed is True
    assert failures == []
    assert calls
    assert all(call["repetition_penalty"] == 1.2 for call in calls)
    assert all(call["min_p"] == 0.05 for call in calls)
    assert all(call["top_k"] == 64 for call in calls)
    assert all(call["top_p"] == 1.0 for call in calls)
    assert all(call["temperature"] == 0.0 for call in calls)


def test_collapse_probe_suite_records_raw_warning_without_failing(monkeypatch) -> None:
    call_index = 0

    def fake_generate(**kwargs):
        nonlocal call_index
        call_index += 1
        if call_index <= len(collapse_probe.SENTINEL_PROMPTS):
            return _Result("\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n\\end{tikzpicture}")
        return _Result("\\calc(0,0) " * 80)

    monkeypatch.setattr(collapse_probe, "generate", fake_generate)

    payload = collapse_probe.run_collapse_probe_suite(
        _Model(),
        processor=object(),
        build_prompt_fn=lambda text: text,
    )

    assert payload["passed"] is True
    assert payload["production"]["passed"] is True
    assert payload["raw_greedy_warning"]["passed"] is False
    assert payload["raw_greedy_warning"]["warning_only"] is True
    assert payload["forced_prefix_diagnostic"] is None


def test_collapse_probe_generation_exception_fails_probe_and_restores_train(monkeypatch) -> None:
    model = _Model()

    def fake_generate(**kwargs):
        raise ValueError("generation failed")

    monkeypatch.setattr(collapse_probe, "generate", fake_generate)

    passed, failures = collapse_probe.run_collapse_probe(
        model,
        processor=object(),
        build_prompt_fn=lambda text: text,
    )

    assert passed is False
    assert model.mode == "train"
    assert failures
    assert failures[0]["response"] == ""
    assert "probe_generation_exception: ValueError: generation failed" in failures[0]["reasons"]


def test_collapse_probe_suite_records_forced_prefix_diagnostic_after_failure(monkeypatch) -> None:
    prompts = []

    def fake_generate(*, prompt, **kwargs):
        prompts.append(prompt)
        if prompt.endswith("\\begin{tikzpicture}\n"):
            return _Result("\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n\\end{tikzpicture}")
        return _Result("\\draw (0,0) -- (1,1);\n" * 20)

    monkeypatch.setattr(collapse_probe, "generate", fake_generate)

    payload = collapse_probe.run_collapse_probe_suite(
        _Model(),
        processor=object(),
        build_prompt_fn=lambda text: text,
        forced_prefix="\\begin{tikzpicture}\n",
    )

    assert payload["passed"] is False
    assert payload["forced_prefix_diagnostic"]["diagnostic_only"] is True
    assert payload["forced_prefix_diagnostic"]["passed"] is True
    assert any(prompt.endswith("\\begin{tikzpicture}\n") for prompt in prompts)


def test_collapse_probe_flags_exactly_five_repeated_lines() -> None:
    text = "\n".join([r"\draw (0,0) -- (1,1);" for _ in range(5)])

    reasons = collapse_probe.check_for_collapse(text)

    assert "Repetition loop detected" in reasons


def test_collapse_probe_flags_normalized_draw_arrow_loop() -> None:
    text = "\n".join(
        rf"\draw[->] (0,{i}) -- (1,{i});"
        for i in range(12)
    )

    reasons = collapse_probe.check_for_collapse(text)

    assert any("normalized" in reason or "Command dominance" in reason for reason in reasons)
