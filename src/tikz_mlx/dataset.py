from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .normalization_audit import normalize_with_audit
from .normalize import normalize_for_training_target
from .prompting import build_generation_prompt
from .schemas import Stage2Sample, TikzSample


@dataclass(slots=True)
class DatasetFingerprint:
    dataset_path: str
    line_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "line_count": self.line_count,
            "sha256": self.sha256,
        }


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def extract_tikz_libraries(tex: str) -> list[str]:
    libraries: set[str] = set()
    for match in re.findall(r"\\usetikzlibrary\{([^}]+)\}", tex):
        libraries.update(lib.strip() for lib in match.split(",") if lib.strip())
    return sorted(libraries)


def detect_generation_mode(tex: str, environment: str | None = None) -> str:
    if re.search(r"\\begin\{axis\}", tex, flags=re.IGNORECASE) or re.search(r"\\addplot\b", tex):
        return "pgfplots_axis"
    if environment == "tikz-cd" or re.search(r"\\begin\{tikzcd\}|\\begin\{tikz-cd\}", tex, flags=re.IGNORECASE):
        return "commutative_diagram"
    if re.search(r"\\graph\s*(?:\[|\{)", tex):
        return "graph_nodes"
    if environment == "circuitikz" or re.search(r"\\begin\{circuitikz\}", tex, flags=re.IGNORECASE):
        return "scientific_schematic"
    return "plain_tikz"


def extract_coordinate_bounding_box(tex: str) -> dict[str, float] | None:
    coordinates: list[tuple[float, float]] = []
    for x_text, y_text in re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", tex):
        try:
            coordinates.append((float(x_text), float(y_text)))
        except ValueError:
            continue
    if not coordinates:
        return None
    xs = [x for x, _ in coordinates]
    ys = [y for _, y in coordinates]
    return {
        "min_x": round(min(xs), 2),
        "min_y": round(min(ys), 2),
        "max_x": round(max(xs), 2),
        "max_y": round(max(ys), 2),
    }


def build_geometry_hints(tex: str, *, generation_mode: str) -> dict[str, object]:
    hints: dict[str, object] = {"generation_mode": generation_mode}
    libraries = extract_tikz_libraries(tex)
    if libraries:
        hints["tikz_libraries"] = libraries
    bounding_box = extract_coordinate_bounding_box(tex)
    if bounding_box is not None:
        hints["bounding_box"] = bounding_box
    return hints


def enrich_training_metadata(sample: TikzSample) -> tuple[dict[str, Any], str, dict[str, object]]:
    metadata = dict(sample.metadata)
    mode = detect_generation_mode(sample.normalized_code, sample.environment)
    hints = build_geometry_hints(sample.normalized_code, generation_mode=mode)
    metadata["generation_mode"] = mode
    metadata["geometry_hints"] = hints
    return metadata, mode, hints


def sample_to_training_record(sample: TikzSample) -> dict:
    if not sample.description:
        raise ValueError("Training sample must include a description.")

    metadata, generation_mode, geometry_hints = enrich_training_metadata(sample)
    # Use the new body-only normalization
    body = normalize_for_training_target(sample.normalized_code)
    
    prompt = build_generation_prompt(
        sample.description,
        generation_mode=generation_mode,
        geometry_hints=geometry_hints,
    )
    assistant_text = body.strip() + "\n```\n"

    return {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "images": [],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_text,
                    }
                ],
            },
        ],
        "metadata": metadata,
    }


def sample_to_stage2_record(sample: TikzSample) -> dict:
    if not sample.description:
        raise ValueError("Stage 2 training sample must include a description.")

    metadata, generation_mode, geometry_hints = enrich_training_metadata(sample)
    return {
        "sample_id": sample.sample_id,
        "prompt_text": build_generation_prompt(
            sample.description,
            generation_mode=generation_mode,
            geometry_hints=geometry_hints,
        ),
        "reference_code": sample.normalized_code,
        "reference_image_path": sample.image_path,
        "reference_embedding_path": None,
        "metadata": metadata,
    }


def load_stage2_samples(path: str | Path) -> list[Stage2Sample]:
    samples: list[Stage2Sample] = []
    for record in iter_jsonl(path):
        prompt_text = record.get("prompt_text") or record.get("prompt")
        if not prompt_text:
            raise ValueError("Stage 2 record must include `prompt_text` or `prompt`.")
        samples.append(
            Stage2Sample(
                sample_id=str(record["sample_id"]),
                prompt_text=str(prompt_text),
                reference_code=record.get("reference_code"),
                reference_image_path=record.get("reference_image_path"),
                reference_embedding_path=record.get("reference_embedding_path"),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return samples


def build_manifest(samples: Iterable[TikzSample]) -> dict:
    counts_by_environment: dict[str, int] = {}
    count = 0
    for sample in samples:
        count += 1
        counts_by_environment[sample.environment] = counts_by_environment.get(sample.environment, 0) + 1

    return {
        "total_samples": count,
        "counts_by_environment": counts_by_environment,
    }


def compute_dataset_fingerprint(path: str | Path) -> DatasetFingerprint:
    dataset_path = Path(path).expanduser().resolve()
    line_count = 0
    hasher = hashlib.sha256()
    with dataset_path.open("rb") as handle:
        for line in handle:
            line_count += 1
            hasher.update(line)
    return DatasetFingerprint(
        dataset_path=str(dataset_path),
        line_count=line_count,
        sha256=hasher.hexdigest(),
    )


def extract_example_indices(records: Sequence[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    for index, record in enumerate(records):
        if "example_index" not in record:
            raise ValueError(
                f"Record at row {index} is missing required `example_index` field. "
                "Re-run dataset splitting before strict coverage training."
            )
        value = record["example_index"]
        if not isinstance(value, int):
            raise ValueError(f"Record at row {index} has non-integer `example_index`: {value!r}")
        indices.append(value)
    return indices


def validate_contiguous_example_indices(indices: Sequence[int]) -> None:
    if not indices:
        raise ValueError("Dataset must contain at least one record with `example_index`.")

    expected = list(range(len(indices)))
    if sorted(indices) != expected:
        raise ValueError(
            "`example_index` values must be a contiguous range 0..N-1 without gaps or duplicates."
        )


def validate_row_aligned_example_indices(indices: Sequence[int]) -> None:
    validate_contiguous_example_indices(indices)
    for row_index, example_index in enumerate(indices):
        if row_index != example_index:
            raise ValueError(
                "`example_index` must match dataset row order (record at row "
                f"{row_index} has example_index={example_index})."
            )


def build_example_index_to_row_map(records: Sequence[dict[str, Any]]) -> dict[int, int]:
    indices = extract_example_indices(records)
    validate_contiguous_example_indices(indices)
    mapping: dict[int, int] = {}
    for row_index, example_index in enumerate(indices):
        mapping[example_index] = row_index
    return mapping


def build_epoch_example_order(
    *,
    total_examples: int,
    order_mode: str,
    seed_base: int,
    epoch: int,
) -> list[int]:
    if total_examples <= 0:
        raise ValueError("total_examples must be positive.")
    if order_mode not in {"epoch-shuffle", "fixed"}:
        raise ValueError("order_mode must be one of: epoch-shuffle, fixed")

    order = list(range(total_examples))
    if order_mode == "epoch-shuffle":
        rng = random.Random(seed_base + epoch)
        rng.shuffle(order)
    return order


def compute_example_order_checksum(order: Sequence[int]) -> str:
    hasher = hashlib.sha256()
    hasher.update("\n".join(str(value) for value in order).encode("utf-8"))
    return hasher.hexdigest()
