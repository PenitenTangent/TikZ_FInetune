import json
from pathlib import Path

from PIL import Image
import pytest

from tikz_mlx.prepare import (
    DEFAULT_SFT_SPLIT_SOURCE_NAME,
    DEFAULT_STAGE2_SPLIT_SOURCE_NAME,
    add_local_figure,
    check_hf_dataset_readiness,
    prepare_hf_dataset,
    split_prepared_dataset,
)
from tikz_mlx.settings import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lora_prod.yaml"


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, truncation=False, add_special_tokens=False):
        return list(range(max(1, len(text.split()))))


def _configure_paths(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    config.paths.data_dir = tmp_path / "data"
    config.paths.prepared_dir = tmp_path / "data" / "prepared"
    config.paths.manifests_dir = tmp_path / "data" / "manifests"
    config.paths.outputs_dir = tmp_path / "outputs"
    config.paths.runs_dir = tmp_path / "runs"
    config.paths.cache_dir = tmp_path / ".cache"
    config.training.train_dataset_path = config.paths.prepared_dir / "train.jsonl"
    config.training.val_dataset_path = config.paths.prepared_dir / "val.jsonl"
    config.training.gold_eval_dataset_path = config.paths.prepared_dir / "gold_eval.jsonl"
    config.training.stage2.dataset_path = config.paths.prepared_dir / "train_stage2.jsonl"
    config.training.stage2.val_dataset_path = config.paths.prepared_dir / "val_stage2.jsonl"
    config.training.stage2.gold_eval_dataset_path = config.paths.prepared_dir / "gold_eval_stage2.jsonl"
    config.training.stage2.checkpoint_dir = config.paths.runs_dir / "stage2_checkpoints"
    config.training.stage2.telemetry_path = config.training.stage2.checkpoint_dir / "metrics.jsonl"
    config.training.stage2.reward_cache_dir = config.paths.cache_dir / "stage2" / "references"
    return config


def test_prepare_hf_dataset_writes_sft_and_stage2_records(monkeypatch, tmp_path) -> None:
    config = _configure_paths(tmp_path)
    monkeypatch.setattr("tikz_mlx.prepare._load_training_tokenizer", lambda model_id: _DummyTokenizer())
    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    records = [
        {
            "file_id": "row-1",
            "caption": "triangle figure",
            "vlm_description": "Draw a triangle with labels A, B, and C.",
            "tikz_code": "\\begin{tikzpicture}\\draw (0,0)--(1,0)--(0,1)--cycle;\\end{tikzpicture}",
            "source": "github",
            "png_image": image,
        }
    ]

    def fake_load_dataset(dataset_id, split, streaming=True):
        assert dataset_id == "nllg/DaTikZ-V4"
        assert split == "train"
        assert streaming is True
        return iter(records)

    monkeypatch.setattr("tikz_mlx.prepare._import_dataset_loader", lambda: fake_load_dataset)

    summary = prepare_hf_dataset(config, overwrite=True)
    assert summary.total_written == 1
    assert summary.train_path.exists()
    assert summary.stage2_path.exists()
    assert (config.paths.prepared_dir / "images").exists()

    train_records = [json.loads(line) for line in summary.train_path.read_text(encoding="utf-8").splitlines()]
    stage2_records = [json.loads(line) for line in summary.stage2_path.read_text(encoding="utf-8").splitlines()]
    assert len(train_records) == 1
    assert len(stage2_records) == 1
    assert stage2_records[0]["reference_image_path"].endswith(".png")
    assert train_records[0]["metadata"]["is_truncated"] is False
    assert train_records[0]["metadata"]["token_length"] > 0
    assert train_records[0]["metadata"]["generation_mode"] == "plain_tikz"
    assert "geometry_hints" in train_records[0]["metadata"]
    assert "mode: plain_tikz" in train_records[0]["messages"][0]["content"][0]["text"]
    assert stage2_records[0]["metadata"]["generation_mode"] == "plain_tikz"
    assert summary.manifest_path.exists()
    assert (config.paths.prepared_dir / DEFAULT_SFT_SPLIT_SOURCE_NAME).exists()
    assert (config.paths.prepared_dir / DEFAULT_STAGE2_SPLIT_SOURCE_NAME).exists()
    assert summary.truncated_records == 0
    assert summary.p99_token_length > 0


def test_add_local_figure_appends_records(tmp_path) -> None:
    config = _configure_paths(tmp_path)
    tex_path = tmp_path / "figure.tex"
    tex_path.write_text(
        "\\begin{tikzpicture}\\draw (0,0)--(1,0)--(0,1)--cycle;\\end{tikzpicture}",
        encoding="utf-8",
    )
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)

    from tikz_mlx import prepare as prepare_module

    original_loader = prepare_module._load_training_tokenizer
    prepare_module._load_training_tokenizer = lambda model_id: _DummyTokenizer()
    try:
        summary = add_local_figure(
            config,
            tex_path=tex_path,
            description="Draw a triangle with labels A, B, and C.",
            image_path=image_path,
            source="local-test",
        )
    finally:
        prepare_module._load_training_tokenizer = original_loader

    train_records = summary.train_path.read_text(encoding="utf-8").splitlines()
    stage2_records = summary.stage2_path.read_text(encoding="utf-8").splitlines()
    assert len(train_records) == 1
    assert len(stage2_records) == 1
    assert summary.total_written == 1


def test_prepare_hf_dataset_rejects_missing_description_rows(monkeypatch, tmp_path) -> None:
    config = _configure_paths(tmp_path)
    monkeypatch.setattr("tikz_mlx.prepare._load_training_tokenizer", lambda model_id: _DummyTokenizer())
    records = [
        {
            "file_id": "row-1",
            "caption": "",
            "vlm_description": "",
            "tikz_code": "\\begin{tikzpicture}\\draw (0,0)--(1,0)--(0,1)--cycle;\\end{tikzpicture}",
            "source": "github",
            "png_image": None,
        }
    ]

    def fake_load_dataset(dataset_id, split, streaming=True):
        assert dataset_id == "nllg/DaTikZ-V4"
        assert split == "train"
        assert streaming is True
        return iter(records)

    monkeypatch.setattr("tikz_mlx.prepare._import_dataset_loader", lambda: fake_load_dataset)

    summary = prepare_hf_dataset(config, overwrite=True)
    assert summary.total_written == 0
    assert summary.total_rejected == 1
    assert summary.rejected_reasons["missing_description"] == 1


def test_check_hf_dataset_readiness_reports_usable_ratio(monkeypatch) -> None:
    records = [
        {
            "tikz_code": "\\begin{tikzpicture}\\draw (0,0)--(1,0)--(0,1)--cycle;\\end{tikzpicture}",
            "caption": "triangle",
            "vlm_description": "",
            "png_image": object(),
        },
        {
            "tikz_code": "",
            "caption": "missing code",
        },
        {
            "tikz_code": "\\begin{tikzpicture}\\draw (0,0)--(1,1);\\end{tikzpicture}",
            "caption": "",
            "vlm_description": "",
        },
    ]

    def fake_load_dataset(dataset_id, split, streaming=True):
        assert dataset_id == "nllg/DaTikZ-V4"
        assert split == "train"
        assert streaming is True
        return iter(records)

    monkeypatch.setattr("tikz_mlx.prepare._import_dataset_loader", lambda: fake_load_dataset)

    summary = check_hf_dataset_readiness(sample_limit=3)
    assert summary.checked_records == 3
    assert summary.usable_records == 1
    assert summary.missing_tikz_code == 1
    assert summary.missing_description == 1
    assert summary.records_with_images == 1


def test_split_prepared_dataset_keeps_content_hash_groups_together(tmp_path: Path) -> None:
    config = _configure_paths(tmp_path)
    train_path = config.paths.prepared_dir / "train.jsonl"
    stage2_path = config.training.stage2.dataset_path
    train_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_path.parent.mkdir(parents=True, exist_ok=True)

    train_records = [
        {
            "sample_id": "a1",
            "messages": [],
            "metadata": {"content_hash": "hash-a"},
        },
        {
            "sample_id": "a2",
            "messages": [],
            "metadata": {"content_hash": "hash-a"},
        },
        {
            "sample_id": "b1",
            "messages": [],
            "metadata": {"content_hash": "hash-b"},
        },
    ]
    stage2_records = [
        {"sample_id": "a1", "prompt_text": "p1"},
        {"sample_id": "a2", "prompt_text": "p2"},
        {"sample_id": "b1", "prompt_text": "p3"},
    ]

    train_path.write_text("\n".join(json.dumps(record) for record in train_records) + "\n", encoding="utf-8")
    stage2_path.write_text("\n".join(json.dumps(record) for record in stage2_records) + "\n", encoding="utf-8")

    summary = split_prepared_dataset(
        config,
        val_fraction=0.0,
        gold_eval_fraction=0.0,
        overwrite=True,
    )
    assert summary.total_records == 3
    assert summary.missing_stage2_records == 0

    train_output = [json.loads(line) for line in summary.train_path.read_text(encoding="utf-8").splitlines() if line]
    val_output = [json.loads(line) for line in summary.val_path.read_text(encoding="utf-8").splitlines() if line]
    gold_output = [json.loads(line) for line in summary.gold_eval_path.read_text(encoding="utf-8").splitlines() if line]
    stage2_train_output = [
        json.loads(line) for line in summary.train_stage2_path.read_text(encoding="utf-8").splitlines() if line
    ]
    stage2_val_output = [json.loads(line) for line in summary.val_stage2_path.read_text(encoding="utf-8").splitlines() if line]
    stage2_gold_output = [
        json.loads(line) for line in summary.gold_eval_stage2_path.read_text(encoding="utf-8").splitlines() if line
    ]

    buckets = {
        "train": {record["sample_id"] for record in train_output},
        "val": {record["sample_id"] for record in val_output},
        "gold": {record["sample_id"] for record in gold_output},
    }
    bucket_by_sample = {
        sample_id: bucket_name
        for bucket_name, sample_ids in buckets.items()
        for sample_id in sample_ids
    }
    assert bucket_by_sample["a1"] == bucket_by_sample["a2"]

    for records in (train_output, val_output, gold_output):
        for example_index, record in enumerate(records):
            assert record["example_index"] == example_index
            assert record["metadata"]["example_index"] == example_index

    for sft_records, stage2_records in (
        (train_output, stage2_train_output),
        (val_output, stage2_val_output),
        (gold_output, stage2_gold_output),
    ):
        by_sample = {record["sample_id"]: record["example_index"] for record in sft_records}
        for record in stage2_records:
            assert record["example_index"] == by_sample[record["sample_id"]]
            assert record["metadata"]["example_index"] == by_sample[record["sample_id"]]


def test_split_prepared_dataset_rejects_in_place_explicit_source(tmp_path: Path) -> None:
    config = _configure_paths(tmp_path)
    train_path = config.paths.prepared_dir / "train.jsonl"
    stage2_path = config.training.stage2.dataset_path
    train_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_path.parent.mkdir(parents=True, exist_ok=True)

    train_path.write_text(
        json.dumps({"sample_id": "a1", "messages": [], "metadata": {"content_hash": "hash-a"}}) + "\n",
        encoding="utf-8",
    )
    stage2_path.write_text(json.dumps({"sample_id": "a1", "prompt_text": "p1"}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must differ from output"):
        split_prepared_dataset(
            config,
            train_path=train_path,
            stage2_path=stage2_path,
            val_fraction=0.0,
            gold_eval_fraction=0.0,
            overwrite=True,
        )


def test_split_prepared_dataset_rejects_missing_stage2_pairs(tmp_path: Path) -> None:
    config = _configure_paths(tmp_path)
    source_train = config.paths.prepared_dir / "all_prepared_sft.jsonl"
    source_stage2 = config.paths.prepared_dir / "all_prepared_stage2.jsonl"
    source_train.parent.mkdir(parents=True, exist_ok=True)
    source_stage2.parent.mkdir(parents=True, exist_ok=True)

    source_train.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "a1", "messages": [], "metadata": {"content_hash": "hash-a"}}),
                json.dumps({"sample_id": "a2", "messages": [], "metadata": {"content_hash": "hash-b"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_stage2.write_text(
        json.dumps({"sample_id": "a1", "prompt_text": "p1"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing records"):
        split_prepared_dataset(
            config,
            train_path=source_train,
            stage2_path=source_stage2,
            val_fraction=0.0,
            gold_eval_fraction=0.0,
            overwrite=True,
        )
