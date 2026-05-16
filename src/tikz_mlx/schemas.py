from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CompileStatus(str, Enum):
    SUCCESS = "success"
    RECOVERABLE_ERROR = "recoverable_error"
    FATAL_ERROR = "fatal_error"
    TIMEOUT = "timeout"
    TOOL_MISSING = "tool_missing"


class RepairMode(str, Enum):
    INITIAL_GENERATION = "initial_generation"
    COMPILE_REPAIR = "compile_repair"
    VISUAL_REPAIR = "visual_repair"


@dataclass(slots=True)
class ExtractedBlock:
    environment: str
    text: str
    start: int
    end: int
    parent_environment: str | None = None


@dataclass(slots=True)
class TikzSample:
    sample_id: str
    source: str
    raw_code: str
    normalized_code: str
    environment: str
    description: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Stage2Sample:
    sample_id: str
    prompt_text: str
    reference_code: str | None = None
    reference_image_path: str | None = None
    reference_embedding_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompileSummary:
    status: CompileStatus
    return_code: int | None
    key_errors: list[str]
    line_hints: list[int]
    missing_packages: list[str]
    stdout: str
    stderr: str
    log_text: str
    elapsed_seconds: float
    tex_path: Path | None = None
    pdf_path: Path | None = None
    log_path: Path | None = None
    working_dir: Path | None = None



@dataclass(slots=True)
class GenerationRequest:
    description: str
    image_paths: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt: str
    model_id: str
    image_paths: list[str]


@dataclass(slots=True)
class RefinementAttempt:
    mode: RepairMode
    prompt: str
    generated_code: str
    compile_summary: CompileSummary | None = None
    debug_image_path: Path | None = None


@dataclass(slots=True)
class EvaluationRecord:
    compiled: bool
    token_count: int
    tex_edit_distance: int | None = None
    selfsim: float | None = None
    emd: float | None = None
    attempt_count: int | None = None
