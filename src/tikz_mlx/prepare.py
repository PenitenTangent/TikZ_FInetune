from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .dataset import iter_jsonl, sample_to_stage2_record, sample_to_training_record, write_jsonl
from .filter import build_sample
from .schemas import TikzSample
from .settings import PipelineConfig, ensure_runtime_directories

DEFAULT_DATASET_ID = "nllg/DaTikZ-V4"
DEFAULT_SFT_SPLIT_SOURCE_NAME = "all_prepared_sft.jsonl"
DEFAULT_STAGE2_SPLIT_SOURCE_NAME = "all_prepared_stage2.jsonl"


@dataclass(slots=True)
class PreparationSummary:
    dataset_id: str
    split: str
    total_seen: int
    total_written: int
    total_rejected: int
    total_duplicates: int
    train_path: Path
    stage2_path: Path
    images_dir: Path
    manifest_path: Path
    counts_by_environment: dict[str, int] = field(default_factory=dict)
    counts_by_source: dict[str, int] = field(default_factory=dict)
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    truncated_records: int = 0
    p99_token_length: int = 0
    max_context_tokens: int = 0


@dataclass(slots=True)
class DatasetReadinessSummary:
    dataset_id: str
    split: str
    checked_records: int
    usable_records: int
    missing_tikz_code: int
    missing_description: int
    records_with_images: int


@dataclass(slots=True)
class DatasetSplitSummary:
    source_train_path: Path
    source_stage2_path: Path
    train_path: Path
    val_path: Path
    gold_eval_path: Path
    train_stage2_path: Path
    val_stage2_path: Path | None
    gold_eval_stage2_path: Path | None
    manifest_path: Path
    total_records: int
    train_records: int
    val_records: int
    gold_eval_records: int
    train_stage2_records: int
    val_stage2_records: int
    gold_eval_stage2_records: int
    missing_stage2_records: int
    grouped_keys: int


def check_hf_dataset_readiness(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = "train",
    sample_limit: int = 128,
) -> DatasetReadinessSummary:
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive.")

    load_dataset = _import_dataset_loader()
    dataset = load_dataset(dataset_id, split=split, streaming=True)

    checked_records = 0
    usable_records = 0
    missing_tikz_code = 0
    missing_description = 0
    records_with_images = 0

    for record in dataset:
        if checked_records >= sample_limit:
            break
        checked_records += 1

        raw_code = _extract_raw_code(record)
        if raw_code is None:
            missing_tikz_code += 1
            continue

        description = _select_description(record)
        if not description:
            missing_description += 1
            continue

        usable_records += 1
        if record.get("png_image") is not None:
            records_with_images += 1

    return DatasetReadinessSummary(
        dataset_id=dataset_id,
        split=split,
        checked_records=checked_records,
        usable_records=usable_records,
        missing_tikz_code=missing_tikz_code,
        missing_description=missing_description,
        records_with_images=records_with_images,
    )


def prepare_hf_dataset(
    config: PipelineConfig,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = "train",
    max_samples: int | None = None,
    overwrite: bool = False,
    allowed_sources: set[str] | None = None,
    progress_interval: int = 1000,
) -> PreparationSummary:
    ensure_runtime_directories(config)
    load_dataset = _import_dataset_loader()

    train_path = config.paths.prepared_dir / "train.jsonl"
    stage2_path = config.training.stage2.dataset_path
    images_dir = config.paths.prepared_dir / "images"
    manifest_path = config.paths.manifests_dir / "datikz_v4_prepare_manifest.json"

    if overwrite:
        for path in (train_path, stage2_path, manifest_path):
            if path.exists():
                path.unlink()
        if images_dir.exists():
            shutil.rmtree(images_dir)

    existing_hashes = _load_existing_content_hashes(train_path)
    total_seen = 0
    total_written = 0
    total_rejected = 0
    total_duplicates = 0
    counts_by_environment: dict[str, int] = {}
    counts_by_source: dict[str, int] = {}
    rejected_reasons: dict[str, int] = {}
    tokenizer = _load_training_tokenizer(config.model.model_id)
    token_lengths: list[int] = []
    truncated_records = 0

    dataset = load_dataset(dataset_id, split=split, streaming=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    with train_path.open("a", encoding="utf-8") as train_handle, stage2_path.open("a", encoding="utf-8") as stage2_handle:
        for index, record in enumerate(dataset):
            if max_samples is not None and index >= max_samples:
                break

            source_name = str(record.get("source", "unknown"))
            if allowed_sources and source_name not in allowed_sources:
                continue

            total_seen += 1
            raw_code = _extract_raw_code(record)
            if raw_code is None:
                total_rejected += 1
                _increment_reason(rejected_reasons, "missing_tikz_code")
                continue

            description = _select_description(record)
            if not description:
                total_rejected += 1
                _increment_reason(rejected_reasons, "missing_description")
                continue

            record_source = f"{dataset_id}:{split}:{record.get('file_id', total_seen)}"
            sample, decision = build_sample(record_source, raw_code, description, config.dataset)
            if sample is None:
                total_rejected += 1
                for reason in decision.reasons:
                    _increment_reason(rejected_reasons, reason)
                continue

            content_hash = str(sample.metadata.get("content_hash", ""))
            if config.dataset.deduplicate and content_hash in existing_hashes:
                total_duplicates += 1
                continue
            existing_hashes.add(content_hash)

            sample = _hydrate_sample_metadata(sample, record, dataset_id, split)
            image_path = images_dir / f"{sample.sample_id}.png"
            if _save_dataset_image(record.get("png_image"), image_path):
                sample.image_path = str(image_path)

            train_record = sample_to_training_record(sample)
            token_length, is_truncated = _annotate_training_record_context(
                train_record,
                tokenizer=tokenizer,
                max_context_tokens=config.model.max_context_tokens,
            )
            token_lengths.append(token_length)
            truncated_records += int(is_truncated)
            stage2_record = sample_to_stage2_record(sample)
            train_handle.write(json.dumps(train_record, ensure_ascii=True) + "\n")
            stage2_handle.write(json.dumps(stage2_record, ensure_ascii=True) + "\n")

            total_written += 1
            counts_by_environment[sample.environment] = counts_by_environment.get(sample.environment, 0) + 1
            counts_by_source[source_name] = counts_by_source.get(source_name, 0) + 1
            if progress_interval > 0 and total_seen % progress_interval == 0:
                print(
                    f"Prepared {total_written} samples after scanning {total_seen} records "
                    f"({total_rejected} rejected, {total_duplicates} duplicates).",
                    flush=True,
                )

    summary = PreparationSummary(
        dataset_id=dataset_id,
        split=split,
        total_seen=total_seen,
        total_written=total_written,
        total_rejected=total_rejected,
        total_duplicates=total_duplicates,
        train_path=train_path,
        stage2_path=stage2_path,
        images_dir=images_dir,
        manifest_path=manifest_path,
        counts_by_environment=counts_by_environment,
        counts_by_source=counts_by_source,
        rejected_reasons=rejected_reasons,
        truncated_records=truncated_records,
        p99_token_length=_percentile_token_length(token_lengths, 0.99),
        max_context_tokens=config.model.max_context_tokens,
    )
    _write_manifest(summary)
    if total_written > 0:
        truncation_rate = truncated_records / total_written
        print(
            "Truncated: "
            f"{truncated_records:,} / {total_written:,} ({truncation_rate * 100:.2f}%) "
            f"- p99 token length: {summary.p99_token_length:,}",
            flush=True,
        )
    _sync_default_split_sources(config, train_path, stage2_path)
    return summary


def split_prepared_dataset(
    config: PipelineConfig,
    *,
    train_path: str | Path | None = None,
    stage2_path: str | Path | None = None,
    val_fraction: float = 0.1,
    gold_eval_fraction: float = 0.05,
    overwrite: bool = False,
) -> DatasetSplitSummary:
    ensure_runtime_directories(config)

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the range [0, 1).")
    if not 0.0 <= gold_eval_fraction < 1.0:
        raise ValueError("gold_eval_fraction must be in the range [0, 1).")
    if val_fraction + gold_eval_fraction >= 1.0:
        raise ValueError("val_fraction + gold_eval_fraction must be less than 1.0.")

    default_source_train = config.paths.prepared_dir / DEFAULT_SFT_SPLIT_SOURCE_NAME
    default_source_stage2 = config.paths.prepared_dir / DEFAULT_STAGE2_SPLIT_SOURCE_NAME

    explicit_source_train = train_path is not None
    explicit_source_stage2 = stage2_path is not None

    if train_path is not None:
        source_train = Path(train_path)
    elif default_source_train.exists():
        source_train = default_source_train
    else:
        source_train = config.paths.prepared_dir / "train.jsonl"

    if stage2_path is not None:
        source_stage2 = Path(stage2_path)
    elif default_source_stage2.exists():
        source_stage2 = default_source_stage2
    else:
        source_stage2 = config.training.stage2.dataset_path

    if not source_train.exists():
        raise RuntimeError(f"Prepared SFT dataset does not exist: {source_train}")
    if not source_stage2.exists():
        raise RuntimeError(f"Prepared stage-2 dataset does not exist: {source_stage2}")

    train_output = config.training.train_dataset_path
    val_output = config.training.val_dataset_path or (config.paths.prepared_dir / "val.jsonl")
    gold_output = config.training.gold_eval_dataset_path or (config.paths.prepared_dir / "gold_eval.jsonl")
    stage2_train_output = config.training.stage2.dataset_path
    stage2_val_output = config.training.stage2.val_dataset_path
    stage2_gold_output = config.training.stage2.gold_eval_dataset_path
    manifest_path = config.paths.manifests_dir / "prepared_split_manifest.json"

    output_paths = [
        train_output,
        val_output,
        gold_output,
        stage2_train_output,
        manifest_path,
    ]
    if stage2_val_output is not None:
        output_paths.append(stage2_val_output)
    if stage2_gold_output is not None:
        output_paths.append(stage2_gold_output)

    if not overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise RuntimeError(
                f"Split outputs already exist: {joined}. Re-run with overwrite=True to replace them."
            )

    source_train = source_train.expanduser().resolve()
    source_stage2 = source_stage2.expanduser().resolve()
    train_output = train_output.expanduser().resolve()
    stage2_train_output = stage2_train_output.expanduser().resolve()

    if source_train == train_output:
        if explicit_source_train:
            raise RuntimeError(
                "split-dataset source train path must differ from output train path. "
                "Use an immutable source file, for example data/prepared/all_prepared_sft.jsonl."
            )
        source_train = _materialize_split_source_snapshot(train_output, default_source_train)

    if source_stage2 == stage2_train_output:
        if explicit_source_stage2:
            raise RuntimeError(
                "split-dataset source stage2 path must differ from output stage2 train path. "
                "Use an immutable source file, for example data/prepared/all_prepared_stage2.jsonl."
            )
        source_stage2 = _materialize_split_source_snapshot(stage2_train_output, default_source_stage2)

    sft_records = list(iter_jsonl(source_train))
    stage2_by_sample_id = {
        str(record.get("sample_id")): record
        for record in iter_jsonl(source_stage2)
        if record.get("sample_id") is not None
    }

    sft_buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "gold_eval": []}
    stage2_buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "gold_eval": []}
    group_assignments: dict[str, str] = {}
    missing_stage2_records = 0

    for record in sft_records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise RuntimeError("Prepared SFT records must include sample_id.")

        metadata = dict(record.get("metadata", {}))
        content_hash = metadata.get("content_hash")
        origin_key = _select_split_origin_key(record)
        if origin_key:
            split_key = f"origin:{origin_key}"
        elif isinstance(content_hash, str) and content_hash:
            split_key = f"content_hash:{content_hash}"
        else:
            split_key = f"sample_id:{sample_id}"
        bucket = group_assignments.get(split_key)
        if bucket is None:
            bucket = _assign_split_bucket(
                split_key,
                split_seed=config.dataset.split_seed,
                val_fraction=val_fraction,
                gold_eval_fraction=gold_eval_fraction,
            )
            group_assignments[split_key] = bucket

        sft_buckets[bucket].append(record)
        paired_stage2 = stage2_by_sample_id.get(sample_id)
        if paired_stage2 is None:
            missing_stage2_records += 1
            continue
        stage2_buckets[bucket].append(paired_stage2)

    if missing_stage2_records > 0:
        raise RuntimeError(
            "Prepared stage-2 dataset is missing records for one or more SFT sample_ids. "
            f"missing_stage2_records={missing_stage2_records}."
        )

    if val_fraction > 0.0 and not sft_buckets["val"]:
        raise RuntimeError(
            "Validation split is empty. Use a larger source dataset or different split fractions/seed."
        )
    if gold_eval_fraction > 0.0 and not sft_buckets["gold_eval"]:
        raise RuntimeError(
            "Gold-eval split is empty. Use a larger source dataset or different split fractions/seed."
        )

    for bucket_name in ("train", "val", "gold_eval"):
        if len(stage2_buckets[bucket_name]) != len(sft_buckets[bucket_name]):
            raise RuntimeError(
                "SFT and stage-2 split bucket sizes diverged for "
                f"bucket={bucket_name}: sft={len(sft_buckets[bucket_name])}, "
                f"stage2={len(stage2_buckets[bucket_name])}."
            )

    for bucket_name in ("train", "val", "gold_eval"):
        _assign_example_indices(
            sft_records=sft_buckets[bucket_name],
            stage2_records=stage2_buckets[bucket_name],
        )

    write_jsonl(train_output, sft_buckets["train"])
    write_jsonl(val_output, sft_buckets["val"])
    write_jsonl(gold_output, sft_buckets["gold_eval"])

    write_jsonl(stage2_train_output, stage2_buckets["train"])
    if stage2_val_output is not None:
        write_jsonl(stage2_val_output, stage2_buckets["val"])
    if stage2_gold_output is not None:
        write_jsonl(stage2_gold_output, stage2_buckets["gold_eval"])

    summary = DatasetSplitSummary(
        source_train_path=source_train,
        source_stage2_path=source_stage2,
        train_path=train_output,
        val_path=val_output,
        gold_eval_path=gold_output,
        train_stage2_path=stage2_train_output,
        val_stage2_path=stage2_val_output,
        gold_eval_stage2_path=stage2_gold_output,
        manifest_path=manifest_path,
        total_records=len(sft_records),
        train_records=len(sft_buckets["train"]),
        val_records=len(sft_buckets["val"]),
        gold_eval_records=len(sft_buckets["gold_eval"]),
        train_stage2_records=len(stage2_buckets["train"]),
        val_stage2_records=len(stage2_buckets["val"]),
        gold_eval_stage2_records=len(stage2_buckets["gold_eval"]),
        missing_stage2_records=missing_stage2_records,
        grouped_keys=len(group_assignments),
    )
    _write_split_manifest(summary, val_fraction=val_fraction, gold_eval_fraction=gold_eval_fraction)
    return summary


def add_local_figure(
    config: PipelineConfig,
    *,
    tex_path: str | Path,
    description: str,
    image_path: str | Path | None = None,
    source: str = "local",
) -> PreparationSummary:
    ensure_runtime_directories(config)

    train_path = config.paths.prepared_dir / "train.jsonl"
    stage2_path = config.training.stage2.dataset_path
    images_dir = config.paths.prepared_dir / "images"
    manifest_path = config.paths.manifests_dir / "datikz_v4_prepare_manifest.json"
    tokenizer = _load_training_tokenizer(config.model.model_id)

    raw_code = Path(tex_path).read_text(encoding="utf-8")
    sample, decision = build_sample(str(Path(tex_path).resolve()), raw_code, description, config.dataset)
    if sample is None:
        reasons = ", ".join(decision.reasons) or "unknown rejection"
        raise RuntimeError(f"Local figure did not pass dataset filters: {reasons}")

    existing_hashes = _load_existing_content_hashes(train_path)
    content_hash = str(sample.metadata.get("content_hash", ""))
    if config.dataset.deduplicate and content_hash in existing_hashes:
        raise RuntimeError("A sample with identical normalized TikZ already exists in the prepared dataset.")

    sample.metadata.update(
        {
            "dataset_id": "local",
            "split": "local",
            "original_source": source,
            "description_source": "manual",
            "file_id": sample.sample_id,
            "environment": sample.environment,
        }
    )

    if image_path is not None:
        images_dir.mkdir(parents=True, exist_ok=True)
        target_image = images_dir / f"{sample.sample_id}{Path(image_path).suffix or '.png'}"
        shutil.copy2(Path(image_path), target_image)
        sample.image_path = str(target_image)

    train_record = sample_to_training_record(sample)
    _annotate_training_record_context(
        train_record,
        tokenizer=tokenizer,
        max_context_tokens=config.model.max_context_tokens,
    )
    stage2_record = sample_to_stage2_record(sample)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("a", encoding="utf-8") as train_handle:
        train_handle.write(json.dumps(train_record, ensure_ascii=True) + "\n")
    with stage2_path.open("a", encoding="utf-8") as stage2_handle:
        stage2_handle.write(json.dumps(stage2_record, ensure_ascii=True) + "\n")

    summary = summarize_prepared_dataset(train_path, stage2_path, config.paths.prepared_dir / "images", manifest_path)
    summary.max_context_tokens = config.model.max_context_tokens
    _write_manifest(summary)
    _sync_default_split_sources(config, train_path, stage2_path)
    return summary


def summarize_prepared_dataset(
    train_path: Path,
    stage2_path: Path,
    images_dir: Path,
    manifest_path: Path,
) -> PreparationSummary:
    counts_by_environment: dict[str, int] = {}
    counts_by_source: dict[str, int] = {}
    total_written = 0
    token_lengths: list[int] = []
    truncated_records = 0
    for record in iter_jsonl(train_path):
        total_written += 1
        metadata = dict(record.get("metadata", {}))
        environment = str(metadata.get("environment", "unknown"))
        source = str(metadata.get("original_source", record.get("source", "unknown")))
        counts_by_environment[environment] = counts_by_environment.get(environment, 0) + 1
        counts_by_source[source] = counts_by_source.get(source, 0) + 1
        token_length = metadata.get("token_length")
        if isinstance(token_length, int):
            token_lengths.append(token_length)
        if bool(metadata.get("is_truncated", False)):
            truncated_records += 1

    return PreparationSummary(
        dataset_id=DEFAULT_DATASET_ID,
        split="train",
        total_seen=total_written,
        total_written=total_written,
        total_rejected=0,
        total_duplicates=0,
        train_path=train_path,
        stage2_path=stage2_path,
        images_dir=images_dir,
        manifest_path=manifest_path,
        counts_by_environment=counts_by_environment,
        counts_by_source=counts_by_source,
        rejected_reasons={},
        truncated_records=truncated_records,
        p99_token_length=_percentile_token_length(token_lengths, 0.99),
        max_context_tokens=0,
    )


def _import_dataset_loader() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The `datasets` package is required for Hugging Face dataset preparation.") from exc
    return load_dataset


def _select_description(record: dict[str, Any]) -> str | None:
    vlm_description = str(record.get("vlm_description", "") or "").strip()
    if vlm_description:
        return vlm_description
    caption = str(record.get("caption", "") or "").strip()
    return caption or None


def _hydrate_sample_metadata(
    sample: TikzSample,
    record: dict[str, Any],
    dataset_id: str,
    split: str,
) -> TikzSample:
    sample.metadata.update(
        {
            "dataset_id": dataset_id,
            "split": split,
            "original_source": str(record.get("source", "unknown")),
            "description_source": "vlm_description" if str(record.get("vlm_description", "") or "").strip() else "caption",
            "file_id": str(record.get("file_id", sample.sample_id)),
            "caption": str(record.get("caption", "") or ""),
            "origin": str(record.get("origin") or record.get("uri") or record.get("file_id") or sample.sample_id),
            "uri": str(record.get("uri", "") or ""),
            "environment": sample.environment,
        }
    )
    return sample


def _save_dataset_image(image: Any, output_path: Path) -> bool:
    if image is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save(output_path)
    except Exception:
        return False
    return True


def _extract_raw_code(record: dict[str, Any]) -> str | None:
    raw_code = record.get("tikz_code")
    if not isinstance(raw_code, str):
        return None
    cleaned = raw_code.strip()
    if not cleaned:
        return None
    return cleaned


def _increment_reason(rejected_reasons: dict[str, int], reason: str) -> None:
    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1


def _load_existing_content_hashes(train_path: Path) -> set[str]:
    if not train_path.exists():
        return set()
    hashes: set[str] = set()
    for record in iter_jsonl(train_path):
        metadata = dict(record.get("metadata", {}))
        content_hash = metadata.get("content_hash")
        if isinstance(content_hash, str) and content_hash:
            hashes.add(content_hash)
    return hashes


def _load_training_tokenizer(model_id: str) -> Any:
    # Use AutoTokenizer directly — mlx_vlm.load_processor always attempts to load
    # a video_preprocessor_config.json that doesn't exist for text-only models,
    # producing a spurious WARNING on every run.
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except ImportError as exc:
        raise RuntimeError("transformers is required to compute truncation statistics.") from exc


def _flatten_messages_for_tokenization(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    flat_messages: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        flat_messages.append(
            {
                "role": str(message.get("role", "")),
                "content": str(content),
            }
        )
    return flat_messages


def _annotate_training_record_context(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    max_context_tokens: int,
) -> tuple[int, bool]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Training record must include non-empty `messages`.")

    text = tokenizer.apply_chat_template(
        _flatten_messages_for_tokenization(messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    token_ids = tokenizer.encode(
        text,
        truncation=False,
        add_special_tokens=False,
    )
    token_length = len(token_ids)
    is_truncated = token_length > max_context_tokens
    metadata = dict(record.get("metadata", {}))
    metadata["token_length"] = token_length
    metadata["is_truncated"] = is_truncated
    record["metadata"] = metadata
    return token_length, is_truncated


def _percentile_token_length(lengths: list[int], quantile: float) -> int:
    if not lengths:
        return 0
    ordered = sorted(lengths)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return int(ordered[index])


def _write_manifest(summary: PreparationSummary) -> None:
    summary.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_id": summary.dataset_id,
        "split": summary.split,
        "total_seen": summary.total_seen,
        "total_written": summary.total_written,
        "total_rejected": summary.total_rejected,
        "total_duplicates": summary.total_duplicates,
        "train_path": str(summary.train_path),
        "stage2_path": str(summary.stage2_path),
        "images_dir": str(summary.images_dir),
        "counts_by_environment": summary.counts_by_environment,
        "counts_by_source": summary.counts_by_source,
        "rejected_reasons": summary.rejected_reasons,
        "truncated_records": summary.truncated_records,
        "p99_token_length": summary.p99_token_length,
        "max_context_tokens": summary.max_context_tokens,
    }
    with summary.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _assign_split_bucket(
    key: str,
    *,
    split_seed: int,
    val_fraction: float,
    gold_eval_fraction: float,
) -> str:
    digest = hashlib.sha1(f"{split_seed}:{key}".encode("utf-8")).digest()
    ratio = int.from_bytes(digest[:8], "big") / float(2**64)
    if ratio < gold_eval_fraction:
        return "gold_eval"
    if ratio < gold_eval_fraction + val_fraction:
        return "val"
    return "train"


def _write_split_manifest(
    summary: DatasetSplitSummary,
    *,
    val_fraction: float,
    gold_eval_fraction: float,
) -> None:
    summary.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_train_path": str(summary.source_train_path),
        "source_stage2_path": str(summary.source_stage2_path),
        "total_records": summary.total_records,
        "train_records": summary.train_records,
        "val_records": summary.val_records,
        "gold_eval_records": summary.gold_eval_records,
        "train_stage2_records": summary.train_stage2_records,
        "val_stage2_records": summary.val_stage2_records,
        "gold_eval_stage2_records": summary.gold_eval_stage2_records,
        "missing_stage2_records": summary.missing_stage2_records,
        "grouped_keys": summary.grouped_keys,
        "val_fraction": val_fraction,
        "gold_eval_fraction": gold_eval_fraction,
        "train_path": str(summary.train_path),
        "val_path": str(summary.val_path),
        "gold_eval_path": str(summary.gold_eval_path),
        "train_stage2_path": str(summary.train_stage2_path),
        "val_stage2_path": str(summary.val_stage2_path) if summary.val_stage2_path is not None else None,
        "gold_eval_stage2_path": (
            str(summary.gold_eval_stage2_path) if summary.gold_eval_stage2_path is not None else None
        ),
    }
    with summary.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _materialize_split_source_snapshot(source_path: Path, snapshot_path: Path) -> Path:
    snapshot = snapshot_path.expanduser().resolve()
    source = source_path.expanduser().resolve()
    if snapshot.exists():
        return snapshot
    if source == snapshot:
        return source
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot)
    return snapshot


def _sync_default_split_sources(config: PipelineConfig, train_path: Path, stage2_path: Path) -> None:
    default_source_train = config.paths.prepared_dir / DEFAULT_SFT_SPLIT_SOURCE_NAME
    default_source_stage2 = config.paths.prepared_dir / DEFAULT_STAGE2_SPLIT_SOURCE_NAME
    default_source_train.parent.mkdir(parents=True, exist_ok=True)
    default_source_stage2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_path, default_source_train)
    shutil.copy2(stage2_path, default_source_stage2)


def _select_split_origin_key(record: dict[str, Any]) -> str | None:
    metadata = dict(record.get("metadata", {}))
    for key in ("origin", "uri", "source_uri", "file_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("origin", "uri", "file_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _assign_example_indices(
    *,
    sft_records: list[dict[str, Any]],
    stage2_records: list[dict[str, Any]],
) -> None:
    index_by_sample_id: dict[str, int] = {}

    for example_index, record in enumerate(sft_records):
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise RuntimeError("Prepared SFT records must include sample_id.")
        record["example_index"] = example_index
        metadata = dict(record.get("metadata", {}))
        metadata["example_index"] = example_index
        record["metadata"] = metadata
        index_by_sample_id[sample_id] = example_index

    for record in stage2_records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise RuntimeError("Prepared stage-2 records must include sample_id.")
        if sample_id not in index_by_sample_id:
            continue
        example_index = index_by_sample_id[sample_id]
        record["example_index"] = example_index
        metadata = dict(record.get("metadata", {}))
        metadata["example_index"] = example_index
        record["metadata"] = metadata
