#!/usr/bin/env python3
# ruff: noqa: E501
r"""Unify TikZ training datasets for consistent model learning.

Normalizes the assistant response in every training example so the model
learns a single, clean output format instead of memorizing arbitrary
boilerplate differences.

Cleaning passes applied to the ASSISTANT response:
  1. Strip inline LaTeX comments (% ...)
  2. Strip wrapper environments (\begin{frame}, \begin{figure}, etc.)
  3. Strip noise commands (\centering, \caption, \label, \noindent, \lipsum)
  4. Strip beamer-specific commands (\only<>, \onslide, \pause, \frametitle)
  5. Convert deprecated \tikzstyle to modern \tikzset{.../.style={...}}
  6. Normalize indentation: tabs -> 2 spaces, collapse mixed indentation
  7. Remove trailing whitespace on every line
  8. Collapse multiple blank lines into one
  9. Remove invisible Unicode (LRM, ZWS, BOM)
  10. Rebuild prompt preamble based on actual package needs

Usage:
  python3 tools/unify_dataset.py \
    --input data/prepared/train_sweep.jsonl \
    --output data/prepared/train_sweep_unified.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tikz_mlx.normalize import PACKAGE_RULES


# ── Constants ───────────────────────────────────────────────────────────────

PREAMBLE_MARKER = "--- Starting Preamble ---"
OUTPUT_CONSTRAINTS = (
    "Output constraints:\n"
    "- Continue from the markdown fence opened below and close it after the TikZ body.\n"
    "- Use the standalone TikZ document class shown in the preamble below.\n"
    "- Keep the preamble minimal; add packages only when required by the chosen environment.\n"
    "- Preserve geometric constraints from the description (coordinates, labels, and relative placement).\n"
    "- Ensure the result is a compilable standalone document."
)

# Wrapper environments to strip (keep contents, remove the wrapping)
_WRAPPER_ENVS = frozenset({
    "frame", "figure", "figure*", "minipage",
    "center", "adjustbox", "table", "table*",
})

# Single-line noise commands to remove entirely
_NOISE_LINE_COMMANDS = (
    r"\centering",
    r"\caption",
    r"\label",
    r"\noindent",
    r"\lipsum",
    r"\vspace",
    r"\hspace",
    r"\frametitle",
    r"\framesubtitle",
)

# Beamer overlay commands to strip
_BEAMER_RE = re.compile(
    r"\\(?:only|onslide|visible|uncover|alt|temporal)"
    r"\s*<[^>]*>\s*\{",
    re.DOTALL,
)

# Invisible Unicode code points to remove
_INVISIBLE_UNICODE = re.compile(
    r"[\u200e\u200f\u200b\u200c\u200d\ufeff\u00ad]"
)

# TikZ-related line starters (signals that real content has begun)
_TIKZ_STARTERS = (
    r"\begin{tikz", r"\tikzset", r"\tikzstyle",
    r"\newcommand", r"\renewcommand", r"\def\\",
    r"\pgf", r"\begin{axis", r"\begin{circuitikz",
    r"\begin{scope", r"\begin{tikz-cd",
)


# ── Cleaning functions ──────────────────────────────────────────────────────


def _strip_inline_comments(text: str) -> str:
    """Remove % comments while preserving escaped \\%."""
    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(cleaned.rstrip())
    return "\n".join(lines)


def _strip_wrapper_environments(text: str) -> str:
    """Remove wrapper environments, keeping their inner content."""
    result = text
    for env in _WRAPPER_ENVS:
        # \begin{env}[optional][optional]{optional}
        result = re.sub(
            rf"\\begin\{{{re.escape(env)}\}}"
            rf"(?:\[[^\]]*\])*"
            rf"(?:\{{[^}}]*\}})*\s*",
            "", result,
        )
        result = re.sub(
            rf"\\end\{{{re.escape(env)}\}}\s*",
            "", result,
        )
    return result.strip()


def _strip_noise_commands(text: str) -> str:
    """Remove noise commands like \\centering and \\noindent without dropping the whole line."""
    result = text
    # Remove these commands when they appear at the start of a line (possibly with whitespace)
    for cmd in [r"\\centering", r"\\noindent"]:
        result = re.sub(rf"^[ \t]*{cmd}\s*", "", result, flags=re.MULTILINE)
    
    # Remove full line noise commands
    for cmd in [r"\\caption", r"\\label", r"\\lipsum", r"\\vspace", r"\\hspace", r"\\frametitle", r"\\framesubtitle"]:
        # Match the command and its arguments (up to the next newline) if it starts the line
        result = re.sub(rf"^[ \t]*{cmd}(?:\[[^\]]*\])?(?:\{{[^}}]*\}})?\s*\n?", "", result, flags=re.MULTILINE)
    
    return result


def _strip_pre_tikz_noise(text: str) -> str:
    """Remove prose/boilerplate lines that appear before TikZ content."""
    lines = text.split("\n")
    cleaned: list[str] = []
    in_tikz = False

    for line in lines:
        stripped = line.strip()

        if not in_tikz:
            # Check if this line starts TikZ-related content
            if any(stripped.startswith(s) for s in _TIKZ_STARTERS):
                in_tikz = True
            # Also trigger on backslash commands that are clearly LaTeX drawing
            elif stripped.startswith(r"\draw") or stripped.startswith(r"\node") or stripped.startswith(r"\fill"):
                in_tikz = True

        if in_tikz:
            cleaned.append(line)
        else:
            # Before TikZ content: skip prose and noise
            if not stripped:
                continue
            if re.match(r"^[A-Z]|^Some |^The |^A |^An |^In |^This ", stripped):
                continue  # Skip prose sentences
            # Keep anything that looks like LaTeX code
            if stripped.startswith("\\"):
                cleaned.append(line)

    return "\n".join(cleaned).strip() if cleaned else text.strip()


def _strip_beamer_overlays(text: str) -> str:
    r"""Remove beamer overlay specifications like \only<2>{...} -> ..."""
    # Simple case: \only<N>{content} -> content
    # We handle the simple single-brace case
    result = re.sub(
        r"\\(?:only|onslide|visible|uncover)\s*<[^>]*>\s*\{",
        "{",
        text,
    )
    # \pause command (standalone)
    result = re.sub(r"\\pause\b\s*", "", result)
    return result


def _convert_tikzstyle_to_tikzset(text: str) -> str:
    r"""Convert deprecated \tikzstyle to \tikzset."""
    def _replace(m: re.Match) -> str:
        name = m.group(1)
        op = m.group(2).strip()
        content = m.group(3)
        style_type = "/.append style=" if op == "+=" else "/.style="
        return rf"\tikzset{{{name}{style_type}{{{content}}}}}"

    # \tikzstyle{name} = [content] or \tikzstyle{name} += [content]
    result = re.sub(
        r"\\tikzstyle\{([^}]+)\}\s*(\+?=)\s*\[([^\]]+)\]",
        _replace, text,
    )
    # \tikzstyle{name} = {content} or \tikzstyle{name} += {content}
    result = re.sub(
        r"\\tikzstyle\{([^}]+)\}\s*(\+?=)\s*\{([^}]+)\}",
        _replace, result,
    )
    return result


def _normalize_indentation(text: str) -> str:
    """Convert tabs to 2 spaces for consistent indentation."""
    return text.replace("\t", "  ")


def _strip_trailing_whitespace(text: str) -> str:
    """Remove trailing whitespace from every line."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _collapse_blank_lines(text: str) -> str:
    """Collapse runs of 3+ blank lines into a single blank line."""
    return re.sub(r"\n\s*\n\s*\n", "\n\n", text)


def _remove_invisible_unicode(text: str) -> str:
    """Remove invisible Unicode characters (LRM, ZWS, BOM, soft hyphen)."""
    return _INVISIBLE_UNICODE.sub("", text)


def clean_assistant_body(raw_body: str) -> str:
    """Apply all cleaning transformations in the correct order."""
    text = _strip_inline_comments(raw_body)
    text = _remove_invisible_unicode(text)
    text = _strip_wrapper_environments(text)
    text = _strip_noise_commands(text)
    text = _strip_pre_tikz_noise(text)
    text = _strip_beamer_overlays(text)
    text = _convert_tikzstyle_to_tikzset(text)
    text = _normalize_indentation(text)
    text = _strip_trailing_whitespace(text)
    text = _collapse_blank_lines(text)
    return text.strip()


# ── Preamble detection ──────────────────────────────────────────────────────


def _detect_preamble_for_body(body: str) -> str:
    """Build minimal consistent preamble from body content."""
    lines = [r"\documentclass[tikz]{standalone}", r"\usepackage{tikz}"]
    for pattern, package in PACKAGE_RULES:
        if re.search(pattern, body):
            lines.append(package)
    if r"\usepackage{pgfplots}" in "\n".join(lines):
        lines.append(r"\pgfplotsset{compat=1.18}")
    lines.append(r"\begin{document}")
    return "\n".join(lines)


def _strip_legacy_output_constraints(text: str) -> str:
    marker = text.find("\nOutput constraints:")
    if marker >= 0:
        return text[:marker].rstrip()
    return text.rstrip()


def _build_contract_prompt(description_part: str, preamble: str) -> str:
    return (
        f"{_strip_legacy_output_constraints(description_part)}\n\n"
        f"{OUTPUT_CONSTRAINTS}\n\n"
        f"{PREAMBLE_MARKER}\n"
        f"{preamble}\n"
        "```latex"
    )


# ── Record transformation ──────────────────────────────────────────────────


def _extract_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
    return ""


def _set_text(msg: dict, new_text: str) -> dict:
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = new_text
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = new_text
                break
    return msg


def unify_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Transform a single training record to unified format."""
    messages = record.get("messages", [])
    if len(messages) < 2:
        return None

    user_msg = messages[0]
    asst_msg = messages[1]
    user_text = _extract_text(user_msg)
    asst_text = _extract_text(asst_msg)

    if not user_text or not asst_text:
        return None

    # Strip fences if present
    body = asst_text
    had_fences = False
    fence_match = re.match(r"^```(?:latex)?\s*(.*?)\s*```\s*$", body, re.DOTALL)
    if fence_match:
        body = fence_match.group(1)
        had_fences = True

    cleaned_body = clean_assistant_body(body)

    # Must still contain a TikZ environment
    if not re.search(r"\\begin\{(?:tikzpicture|tikz-cd|circuitikz|axis)\}", cleaned_body):
        return record  # Don't drop, just skip cleaning

    # The recovery contract always keeps the opening fence in the user prompt
    # and the closing fence in the assistant completion.
    # Strip any trailing closing fence already present (new format from
    # sample_to_training_record ends with \n```\n; fence_match only fires
    # when the body *starts* with ```latex, so we must strip here too).
    cleaned_body = re.sub(r"\s*```\s*$", "", cleaned_body).rstrip()
    new_asst = cleaned_body + "\n```\n"

    # Rebuild prompt preamble
    preamble_marker = user_text.find(PREAMBLE_MARKER)
    if preamble_marker >= 0:
        description_part = user_text[:preamble_marker].rstrip()
        new_preamble = _detect_preamble_for_body(cleaned_body)
        new_prompt = _build_contract_prompt(description_part, new_preamble)
    else:
        # Legacy fallback for records prepared before the marker existed. Use the
        # last documentclass occurrence so mentions inside output constraints do
        # not become the split point.
        preamble_marker = user_text.rfind(r"\documentclass")
        if preamble_marker >= 0:
            description_part = user_text[:preamble_marker].rstrip()
            new_preamble = _detect_preamble_for_body(cleaned_body)
            new_prompt = _build_contract_prompt(description_part, new_preamble)
        else:
            new_prompt = user_text

    new_record = dict(record)
    new_messages = [dict(user_msg), dict(asst_msg)]
    _set_text(new_messages[0], new_prompt)
    _set_text(new_messages[1], new_asst)
    new_record["messages"] = new_messages
    return new_record


# ── Main ────────────────────────────────────────────────────────────────────


def unify_dataset(
    input_path: Path,
    output_path: Path,
    *,
    verbose: bool = True,
) -> dict[str, int]:
    """Process an entire dataset file."""
    stats: dict[str, int] = {
        "total_input": 0,
        "total_output": 0,
        "dropped": 0,
        "modified": 0,
        "unchanged": 0,
        "tikzstyle_converted": 0,
        "frame_stripped": 0,
        "figure_stripped": 0,
        "tabs_normalized": 0,
        "noise_stripped": 0,
        "beamer_stripped": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total_input"] += 1
            record = json.loads(line)

            old_asst = _extract_text(record["messages"][1]) if len(record.get("messages", [])) > 1 else ""

            unified = unify_record(record)
            if unified is None:
                stats["dropped"] += 1
                continue

            new_asst = _extract_text(unified["messages"][1]) if len(unified.get("messages", [])) > 1 else ""

            if old_asst == new_asst:
                stats["unchanged"] += 1
            else:
                stats["modified"] += 1
                if r"\tikzstyle" in old_asst and r"\tikzstyle" not in new_asst:
                    stats["tikzstyle_converted"] += 1
                if r"\begin{frame}" in old_asst and r"\begin{frame}" not in new_asst:
                    stats["frame_stripped"] += 1
                if r"\begin{figure}" in old_asst and r"\begin{figure}" not in new_asst:
                    stats["figure_stripped"] += 1
                if "\t" in old_asst and "\t" not in new_asst:
                    stats["tabs_normalized"] += 1
                if any(cmd in old_asst for cmd in [r"\centering", r"\caption", r"\label", r"\noindent", r"\lipsum"]):
                    stats["noise_stripped"] += 1
                if any(cmd in old_asst for cmd in [r"\only<", r"\onslide", r"\pause"]):
                    stats["beamer_stripped"] += 1

            unified["example_index"] = stats["total_output"]
            records.append(unified)
            stats["total_output"] += 1

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    if verbose:
        changed = stats["modified"]
        total = max(stats["total_output"], 1)
        print(f"\n{'=' * 60}")
        print(f"  Dataset Unification: {input_path.name}")
        print(f"  Input:  {stats['total_input']} records")
        print(f"  Output: {stats['total_output']} records")
        print(f"{'=' * 60}")
        print(f"  Unchanged:           {stats['unchanged']}")
        print(f"  Modified:            {changed} ({changed / total * 100:.1f}%)")
        print(f"    tikzstyle->tikzset:  {stats['tikzstyle_converted']}")
        print(f"    tabs->spaces:        {stats['tabs_normalized']}")
        print(f"    frame stripped:      {stats['frame_stripped']}")
        print(f"    figure stripped:     {stats['figure_stripped']}")
        print(f"    noise stripped:      {stats['noise_stripped']}")
        print(f"    beamer stripped:     {stats['beamer_stripped']}")
        print(f"  Dropped:             {stats['dropped']}")
        print(f"{'=' * 60}\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unify TikZ training dataset for consistent model learning"
    )
    parser.add_argument("--input", required=True, help="Input JSONL dataset")
    parser.add_argument("--output", required=True, help="Output unified JSONL dataset")
    args = parser.parse_args()
    unify_dataset(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
