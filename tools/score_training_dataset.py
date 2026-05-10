#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tikz_mlx.compiler import CompilerService
from tikz_mlx.debug_render import load_render_config, prepare_image_for_reward_encoder, rasterize_pdf
from tikz_mlx.settings import load_config


FENCE_RE = re.compile(r"```(?:latex|tex)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
DESCRIPTION_RE = re.compile(
    r"Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
    r"(?P<description>.*?)\n\nOutput constraints:",
    flags=re.DOTALL,
)


@dataclass(slots=True)
class CompileResult:
    index: int
    record: dict
    compile_ok: bool
    pdf_path: Path | None


class AlignmentScorer(Protocol):
    backend_name: str
    model_name: str

    def score_pairs(self, image_paths: list[Path], descriptions: list[str], *, batch_size: int) -> list[float]:
        """Return one clipped [0, 1] alignment score per image/description pair."""


def _extract_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _extract_description(record: dict) -> str:
    metadata = dict(record.get("metadata", {}))
    for key in ("description", "vlm_description", "caption"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = record.get("messages", [])
    user_text = next((_extract_text(msg) for msg in messages if msg.get("role") == "user"), "")
    match = DESCRIPTION_RE.search(user_text)
    if match:
        return match.group("description").strip()
    return user_text.strip()


def _extract_latex(text: str) -> str:
    matches = FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _reconstruct_document(record: dict) -> str:
    messages = record.get("messages", [])
    user_text = next((_extract_text(msg) for msg in messages if msg.get("role") == "user"), "")
    assistant_text = next((_extract_text(msg) for msg in messages if msg.get("role") == "assistant"), "")
    assistant_code = _extract_latex(assistant_text)
    if assistant_code.lstrip().startswith(r"\documentclass"):
        return assistant_code

    preamble = None
    marker = "--- Starting Preamble ---"
    if marker in user_text:
        after_marker = user_text.split(marker, 1)[1]
        if "```latex" in after_marker:
            preamble = after_marker.split("```latex", 1)[0].strip()

    if preamble:
        body = assistant_code.strip()
        if not body.endswith(r"\end{document}"):
            return f"{preamble}\n{body}\n\\end{{document}}"
        return f"{preamble}\n{body}"

    return assistant_code


def _compile_record(index: int, record: dict, compiler: CompilerService, output_root: Path) -> CompileResult:
    metadata = dict(record.get("metadata", {}))
    if bool(metadata.get("is_truncated", False)):
        metadata["compile_ok"] = False
        metadata["alignment_score"] = None
        metadata["sample_weight"] = 0.0
        record["metadata"] = metadata
        return CompileResult(index=index, record=record, compile_ok=False, pdf_path=None)

    tex = _reconstruct_document(record)
    sample_id = str(record.get("sample_id", f"row_{index:06d}"))
    compile_dir = output_root / sample_id
    summary = compiler.compile_document(tex, output_dir=compile_dir, job_name="candidate")
    compile_ok = summary.pdf_path is not None and Path(summary.pdf_path).exists()
    metadata["compile_ok"] = compile_ok
    metadata["alignment_score"] = None
    metadata["sample_weight"] = 1.0 if compile_ok else 0.0
    record["metadata"] = metadata
    return CompileResult(
        index=index,
        record=record,
        compile_ok=compile_ok,
        pdf_path=Path(summary.pdf_path) if summary.pdf_path is not None else None,
    )


def _clip_unit(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


class SentenceTransformersClipAlignmentScorer:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for image/text alignment scoring. "
                "Install the alignment extra or pass --alignment-backend none."
            ) from exc

        self.backend_name = "sentence-transformers-clip"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def score_pairs(self, image_paths: list[Path], descriptions: list[str], *, batch_size: int) -> list[float]:
        import numpy as np
        from PIL import Image

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())

        image_embeddings = self.model.encode(
            images,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        text_embeddings = self.model.encode(
            descriptions,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if image_embeddings.shape != text_embeddings.shape:
            raise RuntimeError(
                "Alignment embedding shape mismatch: "
                f"image={image_embeddings.shape}, text={text_embeddings.shape}."
            )
        scores = np.sum(image_embeddings * text_embeddings, axis=1)
        return [_clip_unit(float(score)) for score in scores]


class DetikzifySentenceTransformersAlignmentScorer:
    def __init__(self, *, image_model_id: str, text_model_name: str, memory_config):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for detikzify/text alignment scoring. "
                "Install the alignment extra or pass --alignment-backend none."
            ) from exc
        from tikz_mlx.reward.encoder_detikzify import FrozenDetikzifyEncoder

        self.backend_name = "detikzify-sentence-transformers"
        self.model_name = f"{image_model_id} + {text_model_name}"
        self.image_encoder = FrozenDetikzifyEncoder(image_model_id, memory_config)
        self.text_encoder = SentenceTransformer(text_model_name)

    def score_pairs(self, image_paths: list[Path], descriptions: list[str], *, batch_size: int) -> list[float]:
        import numpy as np

        image_vectors: list[np.ndarray] = []
        for image_path in image_paths:
            matrix = np.asarray(self.image_encoder.encode_image(str(image_path)), dtype=np.float32)
            if matrix.size == 0:
                image_vectors.append(np.zeros((0,), dtype=np.float32))
            else:
                image_vectors.append(matrix.mean(axis=0))

        text_vectors = self.text_encoder.encode(
            descriptions,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if len(image_vectors) != len(text_vectors):
            raise RuntimeError("Alignment scorer returned mismatched image/text counts.")

        scores: list[float] = []
        for image_vector, text_vector in zip(image_vectors, text_vectors, strict=True):
            if image_vector.shape != text_vector.shape:
                raise RuntimeError(
                    "Alignment embedding dimension mismatch for detikzify-sentence-transformers backend: "
                    f"image={image_vector.shape}, text={text_vector.shape}. "
                    "Use a text encoder with the same embedding dimension or switch to sentence-transformers-clip."
                )
            image_norm = float(np.linalg.norm(image_vector))
            text_norm = float(np.linalg.norm(text_vector))
            if image_norm <= 0.0 or text_norm <= 0.0:
                scores.append(0.0)
                continue
            scores.append(_clip_unit(float(np.dot(image_vector, text_vector) / (image_norm * text_norm))))
        return scores

    def close(self) -> None:
        self.image_encoder.unload()


def _build_alignment_scorer(args: argparse.Namespace, config) -> AlignmentScorer | None:
    if args.alignment_backend == "none":
        return None
    if args.alignment_backend == "sentence-transformers-clip":
        return SentenceTransformersClipAlignmentScorer(args.alignment_model)
    if args.alignment_backend == "detikzify-sentence-transformers":
        return DetikzifySentenceTransformersAlignmentScorer(
            image_model_id=args.alignment_image_model or config.training.stage2.reward_model_id or config.model.model_id,
            text_model_name=args.alignment_text_model,
            memory_config=config.memory,
        )
    raise RuntimeError(f"Unsupported alignment backend: {args.alignment_backend}")


def _iter_batches(values: list[CompileResult], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _apply_alignment_scores(
    compiled: list[CompileResult],
    *,
    scorer: AlignmentScorer | None,
    config,
    output_root: Path,
    batch_size: int,
    on_alignment_error: str,
) -> None:
    if scorer is None:
        return

    render_config = load_render_config(config.paths.root_dir)
    candidates = [item for item in compiled if item.compile_ok and item.pdf_path is not None]
    for batch in _iter_batches(candidates, batch_size):
        image_paths: list[Path] = []
        descriptions: list[str] = []
        aligned_items: list[CompileResult] = []
        for item in batch:
            sample_id = str(item.record.get("sample_id", f"row_{item.index:06d}"))
            render_dir = output_root / sample_id / "render"
            try:
                rendered = rasterize_pdf(item.pdf_path, render_dir, render_config=render_config)
                image_paths.append(prepare_image_for_reward_encoder(rendered, render_config))
                descriptions.append(_extract_description(item.record))
                aligned_items.append(item)
            except Exception:
                if on_alignment_error == "fail":
                    raise
                metadata = dict(item.record.get("metadata", {}))
                metadata["alignment_score"] = 0.0
                metadata["sample_weight"] = 0.0
                item.record["metadata"] = metadata

        if not aligned_items:
            continue

        try:
            scores = scorer.score_pairs(image_paths, descriptions, batch_size=batch_size)
        except Exception:
            if on_alignment_error == "fail":
                raise
            scores = [0.0 for _ in aligned_items]

        if len(scores) != len(aligned_items):
            raise RuntimeError(
                f"Alignment scorer returned {len(scores)} scores for {len(aligned_items)} records."
            )
        for item, score in zip(aligned_items, scores, strict=True):
            metadata = dict(item.record.get("metadata", {}))
            alignment_score = _clip_unit(float(score))
            metadata["alignment_score"] = alignment_score
            metadata["sample_weight"] = alignment_score
            item.record["metadata"] = metadata


def _weight_histogram(records: list[dict], bins: int = 10) -> dict[str, int]:
    histogram = {f"{idx / bins:.1f}-{(idx + 1) / bins:.1f}": 0 for idx in range(bins)}
    for record in records:
        value = float(dict(record.get("metadata", {})).get("sample_weight", 0.0))
        idx = min(bins - 1, max(0, int(value * bins)))
        key = f"{idx / bins:.1f}-{(idx + 1) / bins:.1f}"
        histogram[key] += 1
    return histogram


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SFT training records with compile and description-image alignment weights.")
    parser.add_argument("--config", default="configs/lora_prod.yaml")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel compile workers.")
    parser.add_argument(
        "--alignment-backend",
        choices=["none", "sentence-transformers-clip", "detikzify-sentence-transformers"],
        default="sentence-transformers-clip",
        help="Alignment backend. Use none for compile-success-only weights.",
    )
    parser.add_argument(
        "--alignment-model",
        default="sentence-transformers/clip-ViT-B-32",
        help="SentenceTransformers CLIP model for the sentence-transformers-clip backend.",
    )
    parser.add_argument(
        "--alignment-image-model",
        help="Image encoder model id for detikzify-sentence-transformers. Defaults to stage2 reward_model_id or model.model_id.",
    )
    parser.add_argument(
        "--alignment-text-model",
        default="all-MiniLM-L6-v2",
        help="Text encoder for detikzify-sentence-transformers. Must match the image embedding dimension.",
    )
    parser.add_argument("--alignment-batch-size", type=int, default=32)
    parser.add_argument("--on-alignment-error", choices=["fail", "zero"], default="fail")
    parser.add_argument("--manifest-output", help="Optional scoring manifest path.")
    parser.add_argument(
        "--compile-cache-dir",
        default="outputs/training_record_scoring",
        help="Directory for temporary compile artifacts.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    compiler = CompilerService(config.compiler)
    scorer = _build_alignment_scorer(args, config)
    input_path = Path(args.input)
    output_path = Path(args.output)
    compile_root = Path(args.compile_cache_dir)
    compile_root.mkdir(parents=True, exist_ok=True)

    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if args.workers <= 1:
        compiled = [
            _compile_record(index, record, compiler, compile_root)
            for index, record in enumerate(records)
        ]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_compile_record, index, record, compiler, compile_root)
                for index, record in enumerate(records)
            ]
            compiled = [future.result() for future in futures]

    try:
        _apply_alignment_scores(
            compiled,
            scorer=scorer,
            config=config,
            output_root=compile_root,
            batch_size=args.alignment_batch_size,
            on_alignment_error=args.on_alignment_error,
        )
    finally:
        close = getattr(scorer, "close", None)
        if callable(close):
            close()

    scored = [item.record for item in sorted(compiled, key=lambda item: item.index)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    compile_ok = sum(1 for record in scored if bool(record.get("metadata", {}).get("compile_ok", False)))
    truncated = sum(1 for record in scored if bool(record.get("metadata", {}).get("is_truncated", False)))
    weighted = [
        float(record.get("metadata", {}).get("sample_weight", 0.0))
        for record in scored
    ]
    nonzero_weights = sum(1 for value in weighted if value > 0.0)
    summary = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "records": len(scored),
        "compile_ok": compile_ok,
        "compile_rate": compile_ok / len(scored) if scored else 0.0,
        "truncated_skipped": truncated,
        "alignment_backend": getattr(scorer, "backend_name", "none"),
        "alignment_model": getattr(scorer, "model_name", None),
        "nonzero_weight_records": nonzero_weights,
        "sample_weight_mean": sum(weighted) / len(weighted) if weighted else 0.0,
        "sample_weight_histogram": _weight_histogram(scored),
    }
    manifest_output = Path(args.manifest_output) if args.manifest_output else output_path.with_suffix(".manifest.json")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
