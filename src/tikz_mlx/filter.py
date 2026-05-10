from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .normalize import contains_external_dependencies, extract_primary_environment, normalize_tikz
from .schemas import TikzSample
from .settings import DatasetConfig


@dataclass(slots=True)
class FilterDecision:
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def stable_sample_id(source: str, normalized_code: str) -> str:
    digest = hashlib.sha256(f"{source}\n{normalized_code}".encode("utf-8")).hexdigest()
    return digest[:16]


def content_hash(normalized_code: str) -> str:
    return hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()


def validate_normalized_code(normalized_code: str, config: DatasetConfig) -> FilterDecision:
    reasons: list[str] = []
    code_len = len(normalized_code)
    if code_len < config.min_chars:
        reasons.append(f"code shorter than {config.min_chars} characters")
    if code_len > config.max_chars:
        reasons.append(f"code longer than {config.max_chars} characters")
    if config.reject_external_dependencies and contains_external_dependencies(normalized_code):
        reasons.append("external dependencies still present after normalization")

    environment = extract_primary_environment(normalized_code)
    if environment is None:
        reasons.append("no supported TikZ environment found")
    elif environment not in config.supported_environments:
        reasons.append(f"unsupported environment: {environment}")

    return FilterDecision(accepted=not reasons, reasons=reasons)


def build_sample(source: str, raw_code: str, description: str | None, config: DatasetConfig) -> tuple[TikzSample | None, FilterDecision]:
    # Early rejection of external dependencies in raw source to prevent prompt/code mismatch
    if config.reject_external_dependencies and contains_external_dependencies(raw_code):
        return None, FilterDecision(accepted=False, reasons=["external dependencies present in raw source"])

    normalized_code = normalize_tikz(raw_code)
    decision = validate_normalized_code(normalized_code, config)
    if not decision.accepted:
        return None, decision

    environment = extract_primary_environment(normalized_code)
    sample = TikzSample(
        sample_id=stable_sample_id(source, normalized_code),
        source=source,
        raw_code=raw_code,
        normalized_code=normalized_code,
        environment=environment or "tikzpicture",
        description=description,
        metadata={"content_hash": content_hash(normalized_code)},
    )
    return sample, decision
