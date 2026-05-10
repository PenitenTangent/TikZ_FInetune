from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .compiler import CompilerService
from .normalize import normalize_tikz
from .schemas import CompileStatus
from .settings import PipelineConfig
from .static_critic import analyze_tikz_static

# ---------------------------------------------------------------------------
# Hard violations the static critic can never permit.
# ---------------------------------------------------------------------------

_CRITIC_HARD_VIOLATIONS = frozenset({
    "unbalanced_braces",
    "unbalanced_brackets",
})

# ---------------------------------------------------------------------------
# Description Quality Filter
# ---------------------------------------------------------------------------

# Minimum description length to be considered informative.
_MIN_DESCRIPTION_WORDS = 5
_MIN_DESCRIPTION_CHARS = 20

# Geometric / structural terms that indicate a meaningful description.
_GEOMETRIC_TERMS = re.compile(
    r"\b(?:circle|rectangle|square|triangle|arrow|node|edge|path|curve|line|"
    r"axis|plot|graph|diagram|label|coordinate|angle|arc|grid|polygon|"
    r"matrix|color|fill|draw|shape|point|segment|vector|map|tree|flow)\b",
    re.IGNORECASE,
)


def score_description_quality(description: str) -> float:
    """Score 0.0–1.0 reflecting how informative a training description is.

    A score below 0.3 indicates the description is too vague to provide a
    meaningful conditioning signal (e.g. "Figure 1", "a diagram").
    """
    if not description or not description.strip():
        return 0.0
    text = description.strip()
    words = text.split()
    if len(words) < _MIN_DESCRIPTION_WORDS or len(text) < _MIN_DESCRIPTION_CHARS:
        return 0.0
    if len(words) > 400:
        return 0.0  # Too long \u2014 likely a scraped paper, wastes token budget.
    score = min(1.0, len(words) / 30.0) * 0.5  # word count component (50%)
    geo_hits = len(_GEOMETRIC_TERMS.findall(text))
    score += min(0.5, geo_hits * 0.1)           # geometric term component (50%)
    return score


def _extract_description_from_record(record: dict[str, Any]) -> str | None:
    """Find the user prompt (description) text in a JSONL record."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        return text
    return None


def _filter_dataset_by_description_quality(
    dataset: list[dict[str, Any]],
    *,
    min_score: float = 0.3,
) -> tuple[list[dict[str, Any]], int]:
    """Drop records whose user description fails the quality threshold."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for record in dataset:
        description = _extract_description_from_record(record)
        if description is None:
            # No description found — keep and let training handle it.
            kept.append(record)
            continue
        if score_description_quality(description) < min_score:
            dropped += 1
        else:
            kept.append(record)
    return kept, dropped


# ---------------------------------------------------------------------------
# Pedagogical Scoring
# ---------------------------------------------------------------------------

_LIBRARY_TIERS: dict[str, float] = {
    "petri":                  3.0,
    "circuits":               3.0,
    "circuits.ee.IEC":        3.0,
    "circuits.logic.US":      3.0,
    "er":                     3.0,
    "datavisualization":      3.0,
    "mindmap":                3.0,
    "spy":                    2.5,
    "fadings":                2.5,
    "decorations.fractals":   2.5,
    "lindenmayersystems":     2.5,
    "3d":                     2.0,
    "perspective":            2.0,
    "hobby":                  2.0,
    "automata":               2.0,
    "intersections":          1.5,
    "through":                1.5,
    "shadows":                1.5,
    "decorations.markings":   1.5,
    "decorations.pathreplacing": 1.5,
}

_CALC_EXPR_RE   = re.compile(r"\(\$[^)]+\$\)")
_NAMED_NODE_RE  = re.compile(r"\\node\s*\(")
_USETIKZLIB_RE  = re.compile(r"\\usetikzlibrary\{([^}]+)\}")
_DRAW_ONLY_RE   = re.compile(r"\\(draw|fill|filldraw|node|path|coordinate|addplot)\b")


def compute_pedagogical_score(completion: str) -> float:
    """Score 0.0–5.0 reflecting how pedagogically valuable a diagram is for training."""
    score = 1.0
    for lib_block_match in _USETIKZLIB_RE.finditer(completion):
        for lib in lib_block_match.group(1).split(","):
            score += _LIBRARY_TIERS.get(lib.strip(), 0.1)
    if _CALC_EXPR_RE.search(completion):
        score += 0.5
    if _NAMED_NODE_RE.search(completion):
        score += 0.3
    all_cmds = _DRAW_ONLY_RE.findall(completion)
    if set(all_cmds) == {"draw"}:
        score *= 0.5
    return min(5.0, max(0.0, score))


def _annotate_pedagogical_score(record: dict[str, Any]) -> dict[str, Any]:
    """Compute and store ``pedagogical_score`` in the record's metadata dict."""
    completion = _extract_reference_code_from_record(record)
    if not completion:
        return record
    import copy
    record = copy.deepcopy(record)
    metadata = dict(record.get("metadata", {}))
    metadata["pedagogical_score"] = compute_pedagogical_score(completion)
    record["metadata"] = metadata
    return record


# ---------------------------------------------------------------------------
# Bounding Box Check (post-compile via pdfinfo)
# ---------------------------------------------------------------------------

_PDFINFO_AVAILABLE: bool | None = None


def _pdfinfo_available() -> bool:
    global _PDFINFO_AVAILABLE
    if _PDFINFO_AVAILABLE is None:
        result = subprocess.run(["which", "pdfinfo"], capture_output=True, text=True)
        _PDFINFO_AVAILABLE = result.returncode == 0
    return _PDFINFO_AVAILABLE


def _get_pdf_dimensions_pt(pdf_path: Path) -> tuple[float, float]:
    """Return (width_pt, height_pt) of the first page. Returns (0, 0) on failure."""
    if not _pdfinfo_available():
        return 0.0, 0.0
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("page size:"):
                parts = line.split()
                if len(parts) >= 5:
                    return float(parts[2]), float(parts[4].rstrip("pts").rstrip())
    except Exception:
        pass
    return 0.0, 0.0


_MIN_DIMENSION_PT = 5.0


def _is_invisible_pdf(pdf_path: Path) -> bool:
    """Return True if the compiled PDF is effectively blank (< 5pt on any side)."""
    w, h = _get_pdf_dimensions_pt(pdf_path)
    if w <= 0.0 and h <= 0.0:
        return False  # pdfinfo unavailable — assume visible
    return w < _MIN_DIMENSION_PT or h < _MIN_DIMENSION_PT


# ---------------------------------------------------------------------------
# Perceptual Deduplication
# ---------------------------------------------------------------------------

def _perceptual_hash_pdf(pdf_path: Path) -> str | None:
    """Compute a perceptual hash of the first page of a compiled PDF.

    Requires ``poppler`` (``pdftoppm``) and ``Pillow``.  Returns None on failure.
    The hash is a 64-bit dhash string that can detect visually identical diagrams
    produced by different TikZ source code (e.g. redundant transforms or scale
    differences).
    """
    try:
        import struct
        from PIL import Image

        # Render first page to PNG at 72 DPI using pdftoppm
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["pdftoppm", "-r", "72", "-l", "1", "-png", str(pdf_path),
                 os.path.join(tmpdir, "page")],
                capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            png_files = sorted(Path(tmpdir).glob("page-*.png"))
            if not png_files:
                return None
            img = Image.open(png_files[0]).convert("L").resize((9, 8), Image.LANCZOS)

        pixels = list(img.getdata())
        bits = 0
        for row in range(8):
            for col in range(8):
                if pixels[row * 9 + col] > pixels[row * 9 + col + 1]:
                    bits |= 1 << (row * 8 + col)
        return format(bits, "016x")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Compiler Cache (SQLite-backed)
# ---------------------------------------------------------------------------

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS compile_cache (
    content_hash TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    normalised   TEXT,
    phash        TEXT,
    created_at   TEXT DEFAULT (datetime('now'))
);
"""


def _open_cache(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path), check_same_thread=False)
    conn.execute(_CACHE_SCHEMA)
    conn.commit()
    return conn


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_lookup(conn: sqlite3.Connection, key: str) -> dict[str, Any] | None:
    """Return cached result dict or None if not cached."""
    row = conn.execute(
        "SELECT status, normalised, phash FROM compile_cache WHERE content_hash = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return {"status": row[0], "normalised": row[1], "phash": row[2]}


def _cache_store(conn: sqlite3.Connection, key: str, status: str,
                 normalised: str | None, phash: str | None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO compile_cache (content_hash, status, normalised, phash)"
        " VALUES (?, ?, ?, ?)",
        (key, status, normalised, phash),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Core helpers (record extraction / mutation)
# ---------------------------------------------------------------------------

def _extract_reference_code_from_record(record: dict[str, Any]) -> str | None:
    """Find the assistant completion text in a JSONL record."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        return text
    return None


def _set_completion_in_record(record: dict[str, Any], new_text: str) -> dict[str, Any]:
    """Return a shallow copy of *record* with the assistant completion replaced."""
    import copy
    record = copy.deepcopy(record)
    messages = record.get("messages")
    if not isinstance(messages, list):
        return record
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = new_text
            return record
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = new_text
                    return record
    return record


# ---------------------------------------------------------------------------
# Static Critic Gate
# ---------------------------------------------------------------------------

def _filter_dataset_by_static_critic(
    dataset: list[dict[str, Any]],
    *,
    max_violations: int,
) -> tuple[list[dict[str, Any]], int]:
    """Drop training records whose assistant completion fails the static critic gate."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for record in dataset:
        completion = _extract_reference_code_from_record(record)
        if completion is None:
            kept.append(record)
            continue
        report = analyze_tikz_static(completion)
        hard_hit = bool(_CRITIC_HARD_VIOLATIONS & set(report.violations))
        too_many = len(report.violations) > max_violations
        if hard_hit or too_many:
            dropped += 1
        else:
            kept.append(record)
    return kept, dropped


# ---------------------------------------------------------------------------
# Compile-and-Repair Pass (parallel, with cache + bbox + perceptual dedup)
# ---------------------------------------------------------------------------

def _compile_repair_dataset(
    dataset: list[dict[str, Any]],
    *,
    config: "PipelineConfig",
    timeout_seconds: float,
    check_bounding_box: bool = True,
    cache_path: Path | None = None,
    perceptual_dedup: bool = True,
    max_workers: int | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Parallel pre-training compile-and-repair pass.

    Args:
        dataset:           Raw JSONL records.
        config:            PipelineConfig.
        timeout_seconds:   Tectonic timeout per sample.
        check_bounding_box: Reject blank PDFs via pdfinfo.
        cache_path:        Optional SQLite cache path. Compiled results are
                           stored/retrieved so reruns skip recompilation.
        perceptual_dedup:  If True, compute perceptual hashes and discard
                           visually identical duplicates (requires pdftoppm + Pillow).

    Returns:
        (repaired_dataset, repaired_count, kept_original_count, invisible_dropped)
    """
    compiler_cfg = dataclasses.replace(
        config.compiler,
        timeout_seconds=int(max(1, timeout_seconds)),
        keep_logs=False,
        keep_intermediates=False,
    )
    compiler = CompilerService(compiler_cfg)
    if not compiler.is_available():
        return dataset, 0, 0, 0

    work_base = Path(tempfile.mkdtemp(prefix="tikz_repair_preflight_"))
    # Use explicit max_workers if provided, otherwise default to 75% of CPU cores.
    if max_workers is not None:
        n_workers = max(1, max_workers)
    else:
        n_workers = max(1, (os.cpu_count() or 1) * 3 // 4)
    total = len(dataset)
    use_bb = check_bounding_box
    use_cache = cache_path is not None

    # SQLite connection — shared across threads (WAL mode for concurrent writes).
    db: sqlite3.Connection | None = None
    if use_cache:
        db = _open_cache(cache_path)  # type: ignore[arg-type]
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")

    flags = []
    if use_bb and _pdfinfo_available():
        flags.append("bbox-check")
    if use_cache:
        flags.append("cache")
    if perceptual_dedup:
        flags.append("perceptual-dedup")
    print(f"[repair_before_training] Starting parallel repair using {n_workers} threads"
          f"{' (' + ', '.join(flags) + ')' if flags else ''}...")

    _db_lock = __import__("threading").Lock()

    def _repair_single_record(
        args: tuple[int, dict[str, Any]],
    ) -> tuple[int, dict[str, Any], bool, bool, str | None]:
        """Return (index, record, was_repaired, is_invisible, phash)."""
        idx, record = args
        completion = _extract_reference_code_from_record(record)
        if not completion:
            return idx, record, False, False, None

        normalised = normalize_tikz(completion)
        cache_key = _content_hash(normalised)

        # ── Cache lookup ───────────────────────────────────────────────────────
        if db is not None:
            with _db_lock:
                cached = _cache_lookup(db, cache_key)
            if cached is not None:
                if cached["status"] == "invisible":
                    return idx, record, False, True, cached.get("phash")
                if cached["status"] == "success" and cached["normalised"]:
                    was_repaired = cached["normalised"] != completion
                    new_record = _set_completion_in_record(record, cached["normalised"])
                    return idx, new_record, was_repaired, False, cached.get("phash")
                # status == "failed" → keep original
                return idx, record, False, False, None

        # ── Compile (pass 1) ───────────────────────────────────────────────────
        work_dir = work_base / f"sample_{idx:06d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        pass1_dir = work_dir / "pass1"
        summary = compiler.compile_document(normalised, output_dir=pass1_dir, job_name="repair")

        if summary.status == CompileStatus.SUCCESS:
            pdf1 = pass1_dir / "repair.pdf"
            # Bounding box check
            if use_bb and pdf1.exists() and _is_invisible_pdf(pdf1):
                phash = _perceptual_hash_pdf(pdf1) if perceptual_dedup else None
                if db is not None:
                    with _db_lock:
                        _cache_store(db, cache_key, "invisible", None, phash)
                return idx, record, False, True, phash
            # Compute perceptual hash for dedup
            phash = _perceptual_hash_pdf(pdf1) if perceptual_dedup and pdf1.exists() else None
            final_text = normalised
            if db is not None:
                with _db_lock:
                    _cache_store(db, cache_key, "success", final_text, phash)
            was_repaired = final_text != completion
            return idx, _set_completion_in_record(record, final_text) if was_repaired else record, was_repaired, False, phash

        # ── Compile (pass 2) ───────────────────────────────────────────────────
        if summary.line_hints:
            lines = normalised.splitlines()
            bad_lines = {ln - 1 for ln in summary.line_hints if 0 < ln <= len(lines)}
            candidate = normalize_tikz("\n".join(ln for i, ln in enumerate(lines) if i not in bad_lines))
            pass2_dir = work_dir / "pass2"
            summary2 = compiler.compile_document(candidate, output_dir=pass2_dir, job_name="repair")
            if summary2.status == CompileStatus.SUCCESS:
                pdf2 = pass2_dir / "repair.pdf"
                if use_bb and pdf2.exists() and _is_invisible_pdf(pdf2):
                    phash = _perceptual_hash_pdf(pdf2) if perceptual_dedup else None
                    if db is not None:
                        with _db_lock:
                            _cache_store(db, cache_key, "invisible", None, phash)
                    return idx, record, False, True, phash
                phash = _perceptual_hash_pdf(pdf2) if perceptual_dedup and pdf2.exists() else None
                if db is not None:
                    with _db_lock:
                        _cache_store(db, cache_key, "success", candidate, phash)
                return idx, _set_completion_in_record(record, candidate), True, False, phash

        # Both passes failed
        if db is not None:
            with _db_lock:
                _cache_store(db, cache_key, "failed", None, None)
        return idx, record, False, False, None

    results: list[tuple[int, dict[str, Any], bool, bool, str | None]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_repair_single_record, (i, r)) for i, r in enumerate(dataset)]
        processed = 0
        heartbeat_interval = max(1, total // 10)
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            processed += 1
            if processed % heartbeat_interval == 0 or processed == total:
                print(f"[repair_before_training] Progress: {processed}/{total} ({processed*100//total}%)")

    results.sort(key=lambda x: x[0])

    # ── Perceptual deduplication pass ─────────────────────────────────────────
    seen_phashes: set[str] = set()
    final_dataset: list[dict[str, Any]] = []
    repaired_count = 0
    invisible_count = 0
    perceptual_dup_count = 0

    for _, record, was_repaired, is_invisible, phash in results:
        if is_invisible:
            invisible_count += 1
            continue
        if perceptual_dedup and phash is not None:
            if phash in seen_phashes:
                perceptual_dup_count += 1
                continue
            seen_phashes.add(phash)
        if was_repaired:
            repaired_count += 1
        final_dataset.append(record)

    kept_original_count = len(final_dataset) - repaired_count

    if db is not None:
        db.close()
    shutil.rmtree(work_base, ignore_errors=True)

    if perceptual_dup_count:
        print(f"[repair_before_training] Perceptual dedup removed {perceptual_dup_count} visually-identical records.")
    if invisible_count:
        print(f"[repair_before_training] Bounding-box check removed {invisible_count} invisible diagrams.")

    return final_dataset, repaired_count, kept_original_count, invisible_count


# ---------------------------------------------------------------------------
# Top-level harden_jsonl_dataset command
# ---------------------------------------------------------------------------

def harden_jsonl_dataset(
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    *,
    max_violations: int = 0,
    timeout_seconds: float = 10.0,
    annotate_pedagogical_scores: bool = True,
    check_bounding_box: bool = True,
    perceptual_dedup: bool = True,
    min_description_quality: float = 0.3,
    cache_path: Path | None = None,
    curriculum_sort: bool = True,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """End-to-end dataset hardening pipeline.

    Stages:
      1. Static Critic Gate — hard syntactic violations.
      2. Description Quality Filter — vague / garbage prompts.
      3. Compile-and-Repair — normalize, compile, bbox-check, cache.
      4. Perceptual Deduplication — visually identical diagrams.
      5. Pedagogical Scoring — annotate each record with complexity score.
      6. Curriculum Sort — order records simple→complex for staged training.
      7. Write output.
    """
    from .dataset import iter_jsonl, write_jsonl

    print(f"Reading dataset from {input_path}...")
    dataset = list(iter_jsonl(input_path))
    total_initial = len(dataset)
    print(f"Loaded {total_initial} records.")

    # ── Stage 1: Static Critic ────────────────────────────────────────────────
    dataset, critic_dropped = _filter_dataset_by_static_critic(
        dataset, max_violations=max_violations
    )
    print(f"Static critic: {len(dataset)} remaining ({critic_dropped} dropped).")

    if not dataset:
        return {"status": "error", "message": "All records dropped by static critic.", "total_initial": total_initial}

    # ── Stage 2: Description Quality Filter ──────────────────────────────────
    dataset, desc_dropped = _filter_dataset_by_description_quality(
        dataset, min_score=min_description_quality
    )
    print(f"Description quality: {len(dataset)} remaining ({desc_dropped} dropped).")

    if not dataset:
        return {"status": "error", "message": "All records dropped by description quality filter.", "total_initial": total_initial}

    # ── Stage 3 & 4: Compile-and-Repair + Perceptual Dedup ───────────────────
    repaired_dataset, repaired_count, kept_original, invisible_dropped = _compile_repair_dataset(
        dataset,
        config=config,
        timeout_seconds=timeout_seconds,
        check_bounding_box=check_bounding_box,
        cache_path=cache_path,
        perceptual_dedup=perceptual_dedup,
        max_workers=max_workers,
    )
    perceptual_dropped = len(dataset) - invisible_dropped - len(repaired_dataset)

    # ── Stage 5: Pedagogical Scoring ─────────────────────────────────────────
    if annotate_pedagogical_scores:
        print("[harden] Annotating pedagogical scores...")
        repaired_dataset = [_annotate_pedagogical_score(r) for r in repaired_dataset]

    # ── Stage 6: Curriculum Sort (simple → complex) ───────────────────────────
    if curriculum_sort and annotate_pedagogical_scores:
        print("[harden] Sorting by pedagogical score (curriculum ordering)...")
        repaired_dataset.sort(
            key=lambda r: r.get("metadata", {}).get("pedagogical_score", 1.0)
        )

    # ── Stage 7: Write output ─────────────────────────────────────────────────
    print(f"Writing hardened dataset to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, repaired_dataset)

    summary = {
        "status": "success",
        "total_initial": total_initial,
        "critic_dropped": critic_dropped,
        "desc_quality_dropped": desc_dropped,
        "invisible_dropped": invisible_dropped,
        "perceptual_dropped": perceptual_dropped,
        "repaired_count": repaired_count,
        "kept_original": kept_original,
        "total_final": len(repaired_dataset),
        "pedagogical_scores_annotated": annotate_pedagogical_scores,
        "curriculum_sorted": curriculum_sort and annotate_pedagogical_scores,
        "bounding_box_check": check_bounding_box and _pdfinfo_available(),
        "perceptual_dedup_active": perceptual_dedup,
        "cache_active": cache_path is not None,
        "output_path": str(output_path),
    }
    return summary
