#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tikz_mlx.compiler import CompilerService
from tikz_mlx.normalize import normalize_tikz
from tikz_mlx.prompting import PROMPT_CONTRACT_VERSION, build_generation_prompt, prompt_template_sha256
from tikz_mlx.schemas import CompileStatus
from tikz_mlx.settings import load_config


DEFAULT_OUTPUT = Path("data/prepared/curriculum/stage0_compile_curriculum.jsonl")
DEFAULT_AUDIT = Path("data/prepared/curriculum/stage0_compile_curriculum_audit.json")

SAFE_COLORS = ("black", "red", "blue", "green", "orange", "purple", "cyan", "magenta", "teal", "gray")
SAFE_STYLES = ("solid", "dashed", "dotted", "thick", "very thick")


@dataclass(frozen=True, slots=True)
class Candidate:
    description: str
    code: str
    category: str
    source: str
    pattern: str | None = None


def _fmt(value: float | int) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _sample_id(index: int, candidate: Candidate) -> str:
    payload = f"{index}\n{candidate.category}\n{candidate.description}\n{candidate.code}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _record(index: int, candidate: Candidate, template_hash: str) -> dict[str, Any]:
    return {
        "example_index": index,
        "messages": [
            {
                "role": "user",
                "content": build_generation_prompt(candidate.description, generation_mode="plain_tikz"),
            },
            {
                "role": "assistant",
                "content": candidate.code.rstrip() + "\n```",
            },
        ],
        "metadata": {
            "example_index": index,
            "generation_mode": "plain_tikz",
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "prompt_template_sha256": template_hash,
            "target_contract": "body_only_environment",
            "stage0_category": candidate.category,
            "stage0_source": candidate.source,
            **({"repair_pattern": candidate.pattern} if candidate.pattern else {}),
        },
        "sample_id": _sample_id(index, candidate),
        "source": candidate.source,
    }


def iter_primitive_candidates(count: int = 900) -> Iterable[Candidate]:
    emitted = 0

    for i in range(180):
        if emitted >= count:
            return
        x1 = (i % 9) - 4
        y1 = ((i * 2) % 9) - 4
        x2 = x1 + 1 + (i % 4) * 0.5
        y2 = y1 + ((i * 3) % 5) - 1
        color = SAFE_COLORS[i % len(SAFE_COLORS)]
        style = SAFE_STYLES[i % len(SAFE_STYLES)]
        desc = f"Draw a {style} {color} segment from ({_fmt(x1)},{_fmt(y1)}) to ({_fmt(x2)},{_fmt(y2)})."
        code = "\n".join(
            [
                "\\begin{tikzpicture}",
                f"\\coordinate (A) at ({_fmt(x1)},{_fmt(y1)});",
                f"\\coordinate (B) at ({_fmt(x2)},{_fmt(y2)});",
                f"\\draw[{style}, {color}] (A) -- (B);",
                "\\end{tikzpicture}",
            ]
        )
        emitted += 1
        yield Candidate(desc, code, "primitive_line", "synthetic:stage0_compile")

    for i in range(150):
        if emitted >= count:
            return
        cx = ((i * 2) % 11) - 5
        cy = ((i * 3) % 9) - 4
        radius = 0.3 + (i % 6) * 0.2
        color = SAFE_COLORS[(i + 2) % len(SAFE_COLORS)]
        label = f"C{i % 10}"
        desc = f"Draw a {color} circle centered at ({_fmt(cx)},{_fmt(cy)}) with radius {_fmt(radius)} and label it {label}."
        code = "\n".join(
            [
                "\\begin{tikzpicture}",
                f"\\coordinate (C) at ({_fmt(cx)},{_fmt(cy)});",
                f"\\draw[{color}] (C) circle ({_fmt(radius)});",
                f"\\node at ({_fmt(cx)},{_fmt(cy)}) {{{label}}};",
                "\\end{tikzpicture}",
            ]
        )
        emitted += 1
        yield Candidate(desc, code, "primitive_circle", "synthetic:stage0_compile")

    for i in range(150):
        if emitted >= count:
            return
        x = (i % 8) - 4
        y = ((i * 3) % 8) - 4
        w = 1 + (i % 4) * 0.5
        h = 0.6 + (i % 5) * 0.4
        color = SAFE_COLORS[(i + 4) % len(SAFE_COLORS)]
        option = f"{color}, rounded corners" if i % 4 == 0 else color
        desc = f"Draw a {color} rectangle from ({_fmt(x)},{_fmt(y)}) to ({_fmt(x + w)},{_fmt(y + h)}) and mark its center."
        code = "\n".join(
            [
                "\\begin{tikzpicture}",
                f"\\coordinate (A) at ({_fmt(x)},{_fmt(y)});",
                f"\\coordinate (B) at ({_fmt(x + w)},{_fmt(y + h)});",
                f"\\draw[{option}] (A) rectangle (B);",
                f"\\node at ({_fmt(x + w / 2)},{_fmt(y + h / 2)}) {{R}};",
                "\\end{tikzpicture}",
            ]
        )
        emitted += 1
        yield Candidate(desc, code, "primitive_rectangle", "synthetic:stage0_compile")

    for i in range(120):
        if emitted >= count:
            return
        cx = (i % 7) - 3
        cy = ((i * 4) % 7) - 3
        rx = 0.8 + (i % 5) * 0.3
        ry = 0.4 + (i % 4) * 0.2
        color = SAFE_COLORS[(i + 6) % len(SAFE_COLORS)]
        desc = f"Draw a {color} ellipse centered at ({_fmt(cx)},{_fmt(cy)}) with x-radius {_fmt(rx)} and y-radius {_fmt(ry)}."
        code = "\n".join(
            [
                "\\begin{tikzpicture}",
                f"\\coordinate (C) at ({_fmt(cx)},{_fmt(cy)});",
                f"\\draw[{color}] (C) ellipse ({_fmt(rx)} and {_fmt(ry)});",
                f"\\node at ({_fmt(cx)},{_fmt(cy)}) {{E}};",
                "\\end{tikzpicture}",
            ]
        )
        emitted += 1
        yield Candidate(desc, code, "primitive_ellipse", "synthetic:stage0_compile")

    for i in range(150):
        if emitted >= count:
            return
        x = (i % 7) - 3
        y = ((i * 2) % 7) - 3
        x2 = x + 2 + (i % 3) * 0.5
        y2 = y + (1 if i % 2 else 0)
        desc = f"Place two labeled nodes and draw an arrow from node A to node B."
        code = "\n".join(
            [
                "\\begin{tikzpicture}",
                f"\\node[draw, circle] (A) at ({_fmt(x)},{_fmt(y)}) {{A}};",
                f"\\node[draw, circle] (B) at ({_fmt(x2)},{_fmt(y2)}) {{B}};",
                "\\draw[->] (A) -- (B);",
                "\\end{tikzpicture}",
            ]
        )
        emitted += 1
        yield Candidate(desc, code, "primitive_nodes", "synthetic:stage0_compile")

    for i in range(90):
        if emitted >= count:
            return
        xmax = 2 + i % 4
        ymax = 2 + (i + 1) % 4
        desc = f"Draw coordinate axes from the origin with x and y arrow labels."
        code = "\n".join(
            [
                "\\begin{tikzpicture}",
                f"\\coordinate (O) at (0,0);",
                f"\\coordinate (X) at ({xmax},0);",
                f"\\coordinate (Y) at (0,{ymax});",
                "\\draw[->] (O) -- (X) node[right] {$x$};",
                "\\draw[->] (O) -- (Y) node[above] {$y$};",
                "\\end{tikzpicture}",
            ]
        )
        emitted += 1
        yield Candidate(desc, code, "primitive_axes", "synthetic:stage0_compile")

    for i in range(60):
        if emitted >= count:
            return
        desc = "Draw a simple three-node flow chart with arrows between the boxes."
        y = (i % 5) * 0.1
        code = "\n".join(
            [
                "\\begin{tikzpicture}",
                f"\\node[draw, rectangle] (A) at (0,{_fmt(y)}) {{Start}};",
                f"\\node[draw, rectangle] (B) at (2,{_fmt(y)}) {{Step}};",
                f"\\node[draw, rectangle] (C) at (4,{_fmt(y)}) {{End}};",
                "\\draw[->] (A) -- (B);",
                "\\draw[->] (B) -- (C);",
                "\\end{tikzpicture}",
            ]
        )
        emitted += 1
        yield Candidate(desc, code, "primitive_flowchart", "synthetic:stage0_compile")


def _repair_templates() -> list[tuple[str, str, str]]:
    return [
        (
            "invalid_style_names",
            "Draw a thick black curved line shaped like a small letter e.",
            "\\begin{tikzpicture}\n"
            "\\coordinate (A) at (0,0);\n"
            "\\coordinate (B) at (2,0.2);\n"
            "\\draw[very thick, black] (A) .. controls (0.4,1.0) and (1.6,1.0) .. (B);\n"
            "\\draw[very thick, black] (1.3,0.2) .. controls (0.8,-0.4) and (0.2,-0.2) .. (0.4,0.4);\n"
            "\\end{tikzpicture}",
        ),
        (
            "safe_color_names",
            "Draw a stacked bar with blue, orange, gray, and dark gray-looking sections.",
            "\\begin{tikzpicture}\n"
            "\\coordinate (A) at (0,0);\n"
            "\\coordinate (B) at (4,0.5);\n"
            "\\fill[blue!30] (0,0) rectangle (1,0.5);\n"
            "\\fill[orange!50] (1,0) rectangle (2,0.5);\n"
            "\\fill[gray!30] (2,0) rectangle (3,0.5);\n"
            "\\fill[gray!70] (3,0) rectangle (4,0.5);\n"
            "\\draw (A) rectangle (B);\n"
            "\\end{tikzpicture}",
        ),
        (
            "named_coordinates_before_use",
            "Draw a triangle ABC with the angle at B highlighted.",
            "\\begin{tikzpicture}\n"
            "\\coordinate (A) at (4,0);\n"
            "\\coordinate (B) at (0,0);\n"
            "\\coordinate (C) at (1.6,2.5);\n"
            "\\draw[thick] (A) -- (B) -- (C) -- cycle;\n"
            "\\draw[green, thick] (0.7,0) arc (0:58:0.7);\n"
            "\\node[right] at (A) {A};\n"
            "\\node[left] at (B) {B};\n"
            "\\node[above] at (C) {C};\n"
            "\\end{tikzpicture}",
        ),
        (
            "safe_hat_label",
            "Draw two labeled points S and S-hat connected by a line.",
            "\\begin{tikzpicture}\n"
            "\\coordinate (S) at (0,0);\n"
            "\\coordinate (Shat) at (2,1);\n"
            "\\fill (S) circle (1.5pt) node[below] {$S$};\n"
            "\\fill (Shat) circle (1.5pt) node[above] {$\\hat{S}$};\n"
            "\\draw[thick] (S) -- (Shat);\n"
            "\\end{tikzpicture}",
        ),
        (
            "explicit_arithmetic_coordinates",
            "Draw three tangent-looking circles inside a triangle using explicit coordinates.",
            "\\begin{tikzpicture}\n"
            "\\coordinate (A) at (0,3);\n"
            "\\coordinate (B) at (-2,0);\n"
            "\\coordinate (C) at (2,0);\n"
            "\\draw[thick] (A) -- (B) -- (C) -- cycle;\n"
            "\\draw[dotted] (-0.8,0.9) circle (0.45);\n"
            "\\draw[dotted] (0.8,0.9) circle (0.45);\n"
            "\\draw[dotted] (0,1.8) circle (0.45);\n"
            "\\node at (-0.8,0.9) {$C_1$};\n"
            "\\node at (0.8,0.9) {$C_2$};\n"
            "\\node at (0,1.8) {$C_3$};\n"
            "\\end{tikzpicture}",
        ),
        (
            "single_tikzpicture_chart",
            "Draw two horizontal mini bar charts in one TikZ picture.",
            "\\begin{tikzpicture}\n"
            "\\node[left] at (0,1.2) {(a)};\n"
            "\\fill[orange!50] (0,1) rectangle (1.2,1.4);\n"
            "\\fill[gray!30] (1.2,1) rectangle (3.6,1.4);\n"
            "\\fill[blue!40] (3.6,1) rectangle (4.4,1.4);\n"
            "\\node[left] at (0,0.2) {(b)};\n"
            "\\fill[orange!50] (0,0) rectangle (1.6,0.4);\n"
            "\\fill[gray!30] (1.6,0) rectangle (3.8,0.4);\n"
            "\\fill[blue!40] (3.8,0) rectangle (4.2,0.4);\n"
            "\\draw[->] (0,-0.3) -- (4.8,-0.3) node[right] {Percent};\n"
            "\\end{tikzpicture}",
        ),
        (
            "no_placeholder_comments",
            "Draw a rectangle, a circle, and one arrow from the rectangle to the circle.",
            "\\begin{tikzpicture}\n"
            "\\node[draw, rectangle] (R) at (0,0) {Box};\n"
            "\\node[draw, circle] (C) at (3,0) {C};\n"
            "\\draw[->, thick] (R) -- (C);\n"
            "\\end{tikzpicture}",
        ),
        (
            "bezier_not_curve_keyword",
            "Draw a smooth curved arrow from node A to node B.",
            "\\begin{tikzpicture}\n"
            "\\node[draw, circle] (A) at (0,0) {A};\n"
            "\\node[draw, circle] (B) at (3,0) {B};\n"
            "\\draw[->, thick] (A) .. controls (1,1.2) and (2,1.2) .. (B);\n"
            "\\end{tikzpicture}",
        ),
        (
            "clean_close_once",
            "Draw a vertical black line and stop after the TikZ environment.",
            "\\begin{tikzpicture}\n"
            "\\coordinate (A) at (0,0);\n"
            "\\coordinate (B) at (0,2);\n"
            "\\draw[black, thick] (A) -- (B);\n"
            "\\end{tikzpicture}",
        ),
        (
            "valid_tree_nodes",
            "Draw a small binary tree with labels at each node.",
            "\\begin{tikzpicture}\n"
            "\\node (X) at (0,2) {$x_1$};\n"
            "\\node (L) at (-1,1) {$x_2$};\n"
            "\\node (R) at (1,1) {$x_2$};\n"
            "\\node (LL) at (-1.5,0) {$0$};\n"
            "\\node (LR) at (-0.5,0) {$0$};\n"
            "\\node (RL) at (0.5,0) {$\\frac{-a_1}{\\sqrt{2}}$};\n"
            "\\node (RR) at (1.5,0) {$\\frac{-a_0}{\\sqrt{2}}$};\n"
            "\\draw (X) -- (L);\n"
            "\\draw (X) -- (R);\n"
            "\\draw (L) -- (LL);\n"
            "\\draw (L) -- (LR);\n"
            "\\draw (R) -- (RL);\n"
            "\\draw (R) -- (RR);\n"
            "\\end{tikzpicture}",
        ),
    ]


def _discover_failure_patterns(outputs_dir: Path) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    dirs: list[str] = []
    for path in sorted(outputs_dir.glob("ab_eval_*/worst_cases.json")):
        dirs.append(str(path.parent))
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            text = str(row.get("response", ""))
            if re.search(r"\b(?:bold|light blue|dark gray|beige|transparent)\b", text):
                counts["invalid_style_names"] += 1
            if "(\\hat" in text or "some_other_node" in text or re.search(r"\([A-Za-z_]+[0-9]*\)", text):
                counts["invalid_coordinates_or_nodes"] += 1
            if "((" in text or " + " in text:
                counts["bad_arithmetic_syntax"] += 1
            if text.count("\\begin{tikzpicture}") > 1:
                counts["nested_tikzpicture"] += 1
            if "Placeholder" in text or "approximation" in text.lower() or "%" in text:
                counts["placeholder_or_commentary"] += 1
            if " curve " in text:
                counts["invalid_path_operation"] += 1
            if text.count("\\end{tikzpicture}") > 1 or text.count("```") > 1:
                counts["repeated_closure_or_fence"] += 1
    return counts, dirs


def iter_repair_candidates(count: int = 400, outputs_dir: Path = Path("outputs")) -> Iterable[Candidate]:
    pattern_counts, _ = _discover_failure_patterns(outputs_dir)
    templates = _repair_templates()
    weighted: list[tuple[str, str, str]] = []
    for pattern, description, code in templates:
        weight = max(1, pattern_counts.get(pattern, 0))
        weighted.extend([(pattern, description, code)] * min(weight, 8))
    if not weighted:
        weighted = templates

    for i in range(count):
        pattern, description, code = weighted[i % len(weighted)]
        suffix = i // len(weighted)
        desc = description
        varied = code
        if suffix:
            shift = (suffix % 5) * 0.1
            varied = varied.replace("(0,0)", f"(0,{_fmt(shift)})", 1)
            desc = f"{description} Use a compact layout variant {suffix}."
        yield Candidate(desc, varied, "repair_pattern", "synthetic:stage0_compile_repair", pattern)


def iter_mixed_candidates(count: int = 200) -> Iterable[Candidate]:
    for i in range(count):
        color = SAFE_COLORS[i % len(SAFE_COLORS)]
        x_shift = (i % 13) * 0.07
        y = ((i // 13) % 17) * 0.05
        width = 3 + (i % 4) * 0.25
        height = 2 + ((i // 4) % 4) * 0.2
        desc = "Draw a compact scene with axes, a labeled circle, a rectangle, and an arrow."
        code = "\n".join(
            [
                "\\begin{tikzpicture}",
                f"\\coordinate (O) at ({_fmt(x_shift)},0);",
                f"\\coordinate (X) at ({_fmt(x_shift + width)},0);",
                f"\\coordinate (Y) at ({_fmt(x_shift)},{_fmt(height)});",
                "\\draw[->] (O) -- (X) node[right] {$x$};",
                "\\draw[->] (O) -- (Y) node[above] {$y$};",
                f"\\node[draw, circle] (C) at ({_fmt(x_shift + 1)},{_fmt(1 + y)}) {{C}};",
                f"\\node[draw, rectangle] (R) at ({_fmt(x_shift + 2.4)},{_fmt(0.7 + y)}) {{R}};",
                f"\\draw[->, {color}] (C) -- (R);",
                "\\end{tikzpicture}",
            ]
        )
        yield Candidate(desc, code, "mixed_composition", "synthetic:stage0_compile_mixed")


def build_candidates(
    *,
    primitive_count: int,
    repair_count: int,
    mixed_count: int,
    outputs_dir: Path,
) -> list[Candidate]:
    return [
        *iter_primitive_candidates(primitive_count),
        *iter_repair_candidates(repair_count, outputs_dir),
        *iter_mixed_candidates(mixed_count),
    ]


def _compile_ok(candidate: Candidate, compiler: CompilerService, tmp_dir: Path, index: int) -> bool:
    source = normalize_tikz(candidate.code)
    job_dir = tmp_dir / f"job_{index:05d}"
    summary = compiler.compile_document(source, output_dir=job_dir, job_name="stage0")
    return summary.status == CompileStatus.SUCCESS


def write_dataset(
    *,
    candidates: list[Candidate],
    output: Path,
    audit_output: Path,
    compiler: CompilerService,
    requested_count: int,
    outputs_dir: Path,
    compile_gate: bool = True,
    compile_workers: int = 1,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    template_hash = prompt_template_sha256()
    rows: list[dict[str, Any]] = []
    rejected_by_compile = 0
    category_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    source_failure_counts, inspected_dirs = _discover_failure_patterns(outputs_dir)

    with tempfile.TemporaryDirectory(prefix="stage0_compile_curriculum_") as tmp_name:
        tmp_dir = Path(tmp_name)
        compile_workers = max(1, int(compile_workers))
        if compile_gate and compile_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=compile_workers) as executor:
                futures = [
                    executor.submit(_compile_ok, candidate, compiler, tmp_dir, index)
                    for index, candidate in enumerate(candidates)
                ]
                for candidate, future in zip(candidates, futures):
                    if len(rows) >= requested_count:
                        break
                    if not future.result():
                        rejected_by_compile += 1
                        continue
                    record = _record(len(rows), candidate, template_hash)
                    rows.append(record)
                    category_counts[candidate.category] += 1
                    if candidate.pattern:
                        pattern_counts[candidate.pattern] += 1
        else:
            for index, candidate in enumerate(candidates):
                if len(rows) >= requested_count:
                    break
                if compile_gate and not _compile_ok(candidate, compiler, tmp_dir, index):
                    rejected_by_compile += 1
                    continue
                record = _record(len(rows), candidate, template_hash)
                rows.append(record)
                category_counts[candidate.category] += 1
                if candidate.pattern:
                    pattern_counts[candidate.pattern] += 1

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    audit = {
        "requested_row_count": requested_count,
        "candidate_count": len(candidates),
        "emitted_row_count": len(rows),
        "rejected_by_compile_count": rejected_by_compile,
        "category_counts": dict(sorted(category_counts.items())),
        "repair_pattern_counts": dict(sorted(pattern_counts.items())),
        "source_failure_pattern_counts": dict(sorted(source_failure_counts.items())),
        "source_ab_eval_dirs_inspected": inspected_dirs,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "prompt_template_sha256": template_hash,
        "output_dataset": str(output),
        "output_dataset_sha256": _file_sha256(output),
        "compile_gate": compile_gate,
        "compile_workers": compile_workers,
    }
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compile-gated Stage 0 syntax curriculum.")
    parser.add_argument("--config", default="configs/curriculum_stage0.yaml")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--target-count", type=int, default=1900)
    parser.add_argument("--primitive-count", type=int, default=900)
    parser.add_argument("--repair-count", type=int, default=400)
    parser.add_argument("--mixed-count", type=int, default=600)
    parser.add_argument("--compile-workers", type=int, default=6)
    parser.add_argument(
        "--no-compile-gate",
        action="store_true",
        help="Write candidates without compiling them. Intended only for tests/debugging.",
    )
    args = parser.parse_args()

    if args.target_count <= 0:
        parser.error("--target-count must be positive")
    config = load_config(args.config)
    compiler = CompilerService(config.compiler)
    if not args.no_compile_gate and not compiler.is_available():
        raise SystemExit(f"Compiler unavailable: {config.compiler.tectonic_binary}")

    candidates = build_candidates(
        primitive_count=args.primitive_count,
        repair_count=args.repair_count,
        mixed_count=args.mixed_count,
        outputs_dir=Path(args.outputs_dir),
    )
    audit = write_dataset(
        candidates=candidates,
        output=Path(args.output),
        audit_output=Path(args.audit_output),
        compiler=compiler,
        requested_count=args.target_count,
        outputs_dir=Path(args.outputs_dir),
        compile_gate=not args.no_compile_gate,
        compile_workers=args.compile_workers,
    )
    print(
        "Built Stage 0 compile curriculum: "
        f"{audit['emitted_row_count']}/{audit['requested_row_count']} rows -> {args.output}"
    )


if __name__ == "__main__":
    main()
