from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


PGFPLOTS_KEY_PATTERN = re.compile(
    r"\b(?:axis\s+[xy]\s+line\*?|xmin|xmax|ymin|ymax|xtick|ytick|xlabel|ylabel|legend\s+(?:entries|pos))\b",
    flags=re.IGNORECASE,
)
HIGH_PRECISION_FLOAT_PATTERN = re.compile(r"-?\d+\.\d{3,}")
COMMAND_PATTERN = re.compile(r"\\[A-Za-z@]+")
COORDINATE_PATTERN = re.compile(r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)")
_COORDINATE_RANGE_LIMIT = 1000.0  # coordinates beyond this are geometric hallucinations


@dataclass(slots=True)
class StaticVerifierReport:
    generation_mode: str | None
    brace_balance: int
    semicolon_rate: float
    num_high_precision_floats: int
    has_tikzpicture: bool
    has_axis: bool
    has_tikzcd: bool
    has_begin_document: bool
    fence_present: bool
    has_pgfplots_keys: bool
    has_legacy_tikzstyle: bool
    command_count: int
    candidate_length: int
    dialect_flags: list[str]
    violations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _balance(text: str, left: str, right: str) -> int:
    return text.count(left) - text.count(right)


def _semicolon_rate(text: str, command_count: int) -> float:
    if command_count <= 0:
        return 0.0
    return min(1.0, text.count(";") / command_count)


def _dialect_flags(text: str) -> list[str]:
    flags: list[str] = []
    if PGFPLOTS_KEY_PATTERN.search(text) or re.search(r"\\addplot\b", text):
        flags.append("pgfplots")
    if re.search(r"\\tikzstyle\b", text):
        flags.append("legacy_tikzstyle")
    if re.search(r"\\graph\s*(?:\[|\{)", text):
        flags.append("graph_syntax")
    if re.search(r"\\begin\{(?:tikzcd|tikz-cd)\}", text, flags=re.IGNORECASE):
        flags.append("tikz_cd")
    return flags


def _is_numeric(s: str) -> bool:
    """Return True if *s* can be parsed as a float (excludes named coords like 'A')."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def analyze_tikz_static(text: str, *, generation_mode: str | None = None) -> StaticVerifierReport:
    # Strip TeX comments to prevent them from causing false-positive structural violations
    # (e.g., an unclosed brace or \begin inside a comment).
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    command_count = len(COMMAND_PATTERN.findall(text))
    has_tikzpicture = bool(re.search(r"\\begin\{tikzpicture\}", text, flags=re.IGNORECASE))
    has_axis = bool(re.search(r"\\begin\{([a-zA-Z]*axis|groupplot)\}", text, flags=re.IGNORECASE))
    has_tikzcd = bool(re.search(r"\\begin\{(?:tikzcd|tikz-cd)\}", text, flags=re.IGNORECASE))
    has_pgfplots_keys = bool(PGFPLOTS_KEY_PATTERN.search(text) or re.search(r"\\addplot\b", text))
    has_legacy_tikzstyle = bool(re.search(r"\\tikzstyle\b", text))
    dialect_flags = _dialect_flags(text)

    violations: list[str] = []
    if _balance(text, "{", "}") != 0:
        violations.append("unbalanced_braces")
    if generation_mode == "plain_tikz" and has_pgfplots_keys and not has_axis:
        violations.append("pgfplots_keys_in_plain_tikz")
    if generation_mode == "pgfplots_axis" and not has_axis:
        violations.append("missing_axis_environment")
    if generation_mode == "commutative_diagram" and not has_tikzcd:
        violations.append("missing_tikzcd_environment")
    if has_legacy_tikzstyle:
        violations.append("legacy_tikzstyle")
    if has_axis and len(re.findall(r"\\begin\{([a-zA-Z]*axis|groupplot)\}", text, flags=re.IGNORECASE)) > 1:
        violations.append("nested_or_duplicate_axis_environment")

    # ── Repetition collapse ──────────────────────────────────────────────────
    # Detects when the model is repeating itself (common failure mode).
    non_empty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(non_empty_lines) > 10:
        most_common = max(set(non_empty_lines), key=non_empty_lines.count)
        if non_empty_lines.count(most_common) > len(non_empty_lines) * 0.5:
            violations.append("repetition_collapse")

    # ── Coordinate out-of-range (geometric hallucination) ───────────────────
    # Skip this check for pgfplots records — axis data coordinates like
    # (5000, 0.95) are data values, not geometric positions.
    coords = COORDINATE_PATTERN.findall(text)
    if not has_axis:
        for x_str, y_str in coords:
            try:
                if abs(float(x_str)) > _COORDINATE_RANGE_LIMIT or abs(float(y_str)) > _COORDINATE_RANGE_LIMIT:
                    violations.append("coordinate_out_of_range")
                    break
            except ValueError:
                pass


    # ── All coordinates at origin (informational — not a hard drop) ──────────
    # NOTE: kept as a warning only; many valid diagrams have a single node at
    # (0,0) and this check is too imprecise to use as a hard filter.
    if coords and all(
        abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6
        for x, y in coords
        if _is_numeric(x) and _is_numeric(y)
    ):
        pass  # informational only — do not append to violations

    # ── Orphan TikZ commands (informational — not a hard drop) ──────────────
    # NOTE: circuitikz and other environments also host \draw commands but are
    # not captured by has_tikzpicture/has_axis/has_tikzcd, so this fires
    # as a false positive too often to use as a hard filter.
    # has_primary_env = has_tikzpicture or has_axis or has_tikzcd
    # if not has_primary_env and re.search(r"\\(?:draw|fill|node|path)\b", text):
    #     violations.append("orphan_tikz_commands")

    # ── Unclosed code fence (training record truncated mid-generation) ────────
    # Only flag when fences are actually used (count >= 1) AND the count is odd.
    # count == 0 means raw LaTeX without markdown formatting, which is valid.
    # The dataset stores only the closing ``` as part of assistant content
    # (count=1), which is the normal format. count==0 is raw LaTeX, also valid.
    # A genuinely truncated record has count >= 3 and odd.
    fence_count = text.count("```")
    if fence_count >= 3 and fence_count % 2 != 0:
        violations.append("unclosed_code_fence")

    # ── Data Quality Guards ──────────────────────────────────────────────────
    if len(text.strip()) < 20:
        violations.append("near_empty_completion")
    
    text_lower = text.lower()
    if "<html" in text_lower or "<?xml" in text_lower or "<div" in text_lower:
        violations.append("html_xml_contamination")
        
    if "import matplotlib" in text_lower or "import numpy" in text_lower or "def plot" in text_lower:
        violations.append("python_code_contamination")
        
    if "current page." in text_lower or "(current page)" in text_lower:
        violations.append("absolute_page_positioning")
        
    if "data:image" in text_lower or re.search(r"[A-Za-z0-9+/]{100,}={0,2}", text):
        violations.append("base64_data_contamination")
        
    if text_lower.count(r"\begin{") != text_lower.count(r"\end{"):
        violations.append("unclosed_environment")
        
    if re.search(r"\\def\\[a-zA-Z]", text) or re.search(r"\\newcommand\{", text):
        pass # informational only \u2014 track frequency of custom macro usage

    return StaticVerifierReport(
        generation_mode=generation_mode,
        brace_balance=_balance(text, "{", "}"),
        semicolon_rate=_semicolon_rate(text, command_count),
        num_high_precision_floats=len(HIGH_PRECISION_FLOAT_PATTERN.findall(text)),
        has_tikzpicture=has_tikzpicture,
        has_axis=has_axis,
        has_tikzcd=has_tikzcd,
        has_begin_document=bool(re.search(r"\\begin\{document\}", text, flags=re.IGNORECASE)),
        fence_present="```" in text,
        has_pgfplots_keys=has_pgfplots_keys,
        has_legacy_tikzstyle=has_legacy_tikzstyle,
        command_count=command_count,
        candidate_length=len(text),
        dialect_flags=dialect_flags,
        violations=violations,
    )
