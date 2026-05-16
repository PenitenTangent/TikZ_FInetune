import inspect
import re
from typing import Any

import mlx.core as mx
from mlx_vlm.generate import generate

SENTINEL_PROMPTS = [
    "Generate a vertical black line.",
    "Draw a red circle centered at (0,0).",
    "Create a simple flow chart with two boxes.",
    "Draw a sine wave from x=0 to x=6."
]

PRODUCTION_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 64,
    "min_p": 0.05,
    "repetition_penalty": 1.2,
}

RAW_GREEDY_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
}

def check_for_collapse(text: str) -> list[str]:
    reasons = []
    if "\\PreviewEnvironment" in text:
        reasons.append("Contains \\PreviewEnvironment")
    if "\\usepackage" in text:
        reasons.append("Contains \\usepackage")
    if "\\documentclass" in text:
        reasons.append("Contains \\documentclass")
    
    # Check for repetition loops (e.g. same line 5 times)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i in range(len(lines) - 5):
        if len(set(lines[i:i+5])) == 1:
            reasons.append("Repetition loop detected")
            break

    # Check for structural repetition via bigram diversity and unigram dominance.
    # A model stuck in a \draw-coordinates loop passes the identical-line check because
    # coordinates differ, but token-level metrics expose the structural repetition.
    tokens = text.split()
    if len(tokens) >= 20:
        # Bigram entropy: catches simple word/symbol repetition loops.
        bigrams = [f"{tokens[i]}|{tokens[i+1]}" for i in range(len(tokens) - 1)]
        unique_ratio = len(set(bigrams)) / len(bigrams)
        if unique_ratio < 0.20:
            reasons.append(f"Low bigram diversity ({unique_ratio:.2f} < 0.20)")

        # Unigram dominance: catches \draw-loop collapse where one command
        # makes up >30% of all tokens (but coordinates differ, fooling bigram check).
        token_counts: dict[str, int] = {}
        for tok in tokens:
            token_counts[tok] = token_counts.get(tok, 0) + 1
        max_token = max(token_counts, key=token_counts.get)
        dominance = token_counts[max_token] / len(tokens)
        # Exclude common structural tokens that are legitimately dominant.
        _ALLOWED_DOMINANT = {r"\draw", "--", ";", ",", "(", ")", "{", "}"}
        if dominance > 0.30 and max_token not in _ALLOWED_DOMINANT:
            reasons.append(f"Unigram dominance: '{max_token}' is {dominance:.0%} of output")

    return reasons

def _supported_generation_kwargs(decoding: dict[str, Any] | None) -> dict[str, Any]:
    if not decoding:
        return {}
    supported = inspect.signature(generate).parameters
    return {key: value for key, value in decoding.items() if key in supported and value is not None}


def run_collapse_probe(model, processor, build_prompt_fn, verbose=False, decoding: dict[str, Any] | None = None):
    """
    Runs a fast probe on the model to see if it has collapsed.
    Returns (passed, list of failures)
    """
    model.eval()
    failures = []
    generation_kwargs = _supported_generation_kwargs(decoding if decoding is not None else PRODUCTION_DECODING)
    
    for prompt_text in SENTINEL_PROMPTS:
        # We need to build the actual prompt using the contract
        full_prompt = build_prompt_fn(prompt_text)
        
        try:
            # Generate a small sample
            result = generate(
                model=model,
                processor=processor,
                prompt=full_prompt,
                max_tokens=512,
                verbose=False,
                **generation_kwargs,
            )
            text = result.text
            reasons = check_for_collapse(text)
            if reasons:
                failures.append({"prompt": prompt_text, "response": text, "reasons": reasons})
        except Exception as e:
            if verbose:
                print(f"Probe failed for prompt '{prompt_text}': {e}")
            continue
            
    model.train()
    return len(failures) == 0, failures


def run_collapse_probe_suite(
    model,
    processor,
    build_prompt_fn,
    verbose: bool = False,
    production_decoding: dict[str, Any] | None = None,
    raw_decoding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    production = production_decoding or PRODUCTION_DECODING
    raw = raw_decoding or RAW_GREEDY_DECODING
    production_passed, production_failures = run_collapse_probe(
        model,
        processor,
        build_prompt_fn,
        verbose=verbose,
        decoding=production,
    )
    raw_passed, raw_failures = run_collapse_probe(
        model,
        processor,
        build_prompt_fn,
        verbose=verbose,
        decoding=raw,
    )
    return {
        "passed": production_passed,
        "production": {
            "passed": production_passed,
            "decoding": production,
            "failures": production_failures,
        },
        "raw_greedy_warning": {
            "passed": raw_passed,
            "decoding": raw,
            "warning_only": True,
            "failures": raw_failures,
        },
        "failures": production_failures,
    }
