from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .schemas import CompileStatus, EvaluationRecord

TOKEN_RE = re.compile(r"\\[A-Za-z@]+|[{}()\[\],;]|-?\d+(?:\.\d+)?|[A-Za-z_]+")


def tokenize_tikz(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def tex_edit_distance(reference: str, candidate: str) -> int:
    left = tokenize_tikz(reference)
    right = tokenize_tikz(candidate)
    if not left:
        return len(right)
    if not right:
        return len(left)

    dp = list(range(len(right) + 1))
    for i, l_token in enumerate(left, start=1):
        previous = dp[0]
        dp[0] = i
        for j, r_token in enumerate(right, start=1):
            old = dp[j]
            if l_token == r_token:
                dp[j] = previous
            else:
                dp[j] = 1 + min(previous, dp[j], dp[j - 1])
            previous = old
    return dp[-1]


def average_token_count(codes: Sequence[str], tokenizer: Callable[[str], Sequence[str]] | None = None) -> float:
    if not codes:
        return 0.0
    tokenize = tokenizer or tokenize_tikz
    total = sum(len(tokenize(code)) for code in codes)
    return total / len(codes)


def compilation_rate(statuses: Sequence[CompileStatus]) -> float:
    if not statuses:
        return 0.0
    compiled = sum(1 for status in statuses if status == CompileStatus.SUCCESS)
    return compiled / len(statuses)


def build_evaluation_record(
    compiled: bool,
    code: str,
    reference_code: str | None = None,
    selfsim: float | None = None,
    emd: float | None = None,
    attempt_count: int | None = None,
) -> EvaluationRecord:
    ted = tex_edit_distance(reference_code, code) if reference_code is not None else None
    return EvaluationRecord(
        compiled=compiled,
        token_count=len(tokenize_tikz(code)),
        tex_edit_distance=ted,
        selfsim=selfsim,
        emd=emd,
        attempt_count=attempt_count,
    )


def summarize_evaluations(records: Sequence[EvaluationRecord]) -> dict[str, float]:
    if not records:
        return {
            "compilation_rate": 0.0,
            "average_tokens": 0.0,
            "average_ted": 0.0,
            "average_attempts": 0.0,
            "average_attempts_to_success": 0.0,
        }

    compiled = sum(1 for record in records if record.compiled)
    average_tokens = sum(record.token_count for record in records) / len(records)
    ted_values = [record.tex_edit_distance for record in records if record.tex_edit_distance is not None]
    attempt_values = [record.attempt_count for record in records if record.attempt_count is not None]
    successful_attempt_values = [
        record.attempt_count for record in records if record.compiled and record.attempt_count is not None
    ]

    return {
        "compilation_rate": compiled / len(records),
        "average_tokens": average_tokens,
        "average_ted": sum(ted_values) / len(ted_values) if ted_values else 0.0,
        "average_attempts": sum(attempt_values) / len(attempt_values) if attempt_values else 0.0,
        "average_attempts_to_success": (
            sum(successful_attempt_values) / len(successful_attempt_values) if successful_attempt_values else 0.0
        ),
    }
