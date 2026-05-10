from __future__ import annotations

import re
import json
import hashlib

from .schemas import CompileSummary

PROMPT_CONTRACT_VERSION = "tikz_body_only_v3"

def stable_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def prompt_template_sha256() -> str:
    # Serialize the actual body-only prompt template so that
    # any change to the prompt text invalidates cached tokenizations.
    body_only_template = (
        "Generate only the TikZ environment body according to the following requirements:\n"
        "{description}\n"
        "{hints}\n"
        "Output constraints:\n"
        "- Generate only the TikZ environment body (e.g., \\begin{tikzpicture} ... \\end{tikzpicture}).\n"
        "- Do not output a LaTeX preamble, \\documentclass, \\usepackage, or \\PreviewEnvironment.\n"
        "- Do not output \\begin{document} or \\end{document}.\n"
        "- Start directly with the TikZ environment and end with the matching close and the markdown fence.\n"
        "- Preserve geometric constraints from the description (coordinates, labels, and relative placement).\n"
        "- Use strict TikZ syntax: terminate paths with ';', use calc ($...$) for math.\n\n"
        "```latex"
    )
    payload = {
        "contract_version": PROMPT_CONTRACT_VERSION,
        "template": body_only_template,
        "rules": [
            "body_only_environment",
            "no_preamble",
            "no_document_wrapper",
            "assistant_starts_with_tikz_env",
            "assistant_ends_with_closing_fence",
        ]
    }
    return stable_json_sha256(payload)


CANONICAL_TIKZ_DOCUMENT_TEMPLATE = (
    "\\documentclass[tikz]{standalone}\n"
    "\\usepackage{tikz}\n"
    "\\begin{document}\n"
    "% choose one environment that matches the requested figure:\n"
    "% \\begin{tikzpicture} ... \\end{tikzpicture}\n"
    "% or \\begin{tikzcd} ... \\end{tikzcd}\n"
    "% or \\begin{circuitikz} ... \\end{circuitikz}\n"
    "\\end{document}\n"
)

DESCRIPTION_PROMPT = (
    "You are a scientific illustrator describing images for precise redrawing in TikZ. "
    "Your task is to describe the image in precise, continuous prose without bullet points, lists, or line breaks. "
    "Start directly with the main object or scene and avoid introductory phrases. "
    "Use clear, active language focused on geometry, labels, colors, spatial relationships, coordinates, dimensions, orientation, and other visible properties. "
    "Describe all visible shapes, lines, arrows, and labels precisely enough for TikZ reconstruction, and avoid vague, interpretive, or aesthetic commentary."
)

DESCRIPTION_REQUEST = (
    "Describe this scientific figure for precise TikZ reconstruction. "
    "Start directly with the main object or scene."
)


def _format_geometry_hints(
    *,
    generation_mode: str | None,
    geometry_hints: dict[str, object] | None,
) -> str:
    lines: list[str] = []
    if generation_mode:
        lines.append(f"mode: {generation_mode}")
    if geometry_hints:
        libraries = geometry_hints.get("tikz_libraries")
        if isinstance(libraries, list) and libraries:
            # Only include if not excessively long to avoid prompt copying
            if len(libraries) <= 6:
                lines.append("tikzlibrary: " + ", ".join(str(value) for value in libraries))
        bounding_box = geometry_hints.get("bounding_box")
        if isinstance(bounding_box, dict):
            try:
                min_x = float(bounding_box["min_x"])
                min_y = float(bounding_box["min_y"])
                max_x = float(bounding_box["max_x"])
                max_y = float(bounding_box["max_y"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                lines.append(f"bounding_box: ({min_x:g}, {min_y:g}) to ({max_x:g}, {max_y:g})")
    if not lines:
        return ""
    return "\n[GEOMETRY HINTS]\n" + "\n".join(lines) + "\n"


def build_generation_prompt(
    description: str,
    *,
    generation_mode: str | None = None,
    geometry_hints: dict[str, object] | None = None,
) -> str:
    hints = _format_geometry_hints(
        generation_mode=generation_mode,
        geometry_hints=geometry_hints,
    )
    return (
        "Generate only the TikZ environment body according to the following requirements:\n"
        f"{description.strip()}\n"
        f"{hints}\n"
        "Output constraints:\n"
        "- Generate only the TikZ environment body (e.g., \\begin{tikzpicture} ... \\end{tikzpicture}).\n"
        "- Do not output a LaTeX preamble, \\documentclass, \\usepackage, or \\PreviewEnvironment.\n"
        "- Do not output a LaTeX document environment (begin/end document).\n"
        "- Start directly with the TikZ environment and end with the matching close and the markdown fence.\n"
        "- Preserve geometric constraints from the description (coordinates, labels, and relative placement).\n"
        "- Use strict TikZ syntax: terminate paths with ';', use calc ($...$) for math.\n\n"
        "```latex"
    )


def build_compile_repair_prompt(code: str, summary: CompileSummary) -> str:
    errors = "\n".join(f"- {entry}" for entry in summary.key_errors) or "- Unknown LaTeX error"
    line_hints = ", ".join(str(line) for line in summary.line_hints) or "none"
    missing_packages = ", ".join(summary.missing_packages) or "none"
    return (
        "Fix the TikZ code so that it compiles without errors while preserving the figure intent.\n"
        "Only output corrected LaTeX code (no Markdown fences, no commentary).\n"
        "Keep edits minimal and avoid changing unrelated parts of the drawing.\n\n"
        f"Key errors:\n{errors}\n"
        f"Line hints: {line_hints}\n"
        f"Missing packages: {missing_packages}\n\n"
        f"Original TikZ code:\n{code}"
    )


def build_visual_repair_prompt(description: str, code: str) -> str:
    return (
        "You are correcting an already-compilable TikZ figure.\n"
        "Use the rendered image with the debug grid to fix geometry, spacing, and coordinates.\n"
        "The grid exists only in the debug image. Do not add grid code or other debug artifacts to the final LaTeX.\n"
        "Preserve valid LaTeX structure, keep edits minimal, and only output corrected LaTeX code.\n"
        "Do not include Markdown fences, explanations, or analysis text.\n\n"
        f"Target description:\n{description.strip()}\n\n"
        f"Current code:\n{code}"
    )


def extract_latex_from_response(text: str) -> str:
    fenced = re.findall(r"```(?:latex)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    return text.strip()


def build_gemma_messages(
    user_text: str,
    image_paths: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content: list[dict[str, str]] = []
    for image_path in image_paths or []:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": content})
    return messages
