import json
import os
from pathlib import Path

import yaml

from tools.run_hparam_sweep import (
    _finish_running_marker,
    _jsonl_has_row_aligned_example_index,
    _materialize_variant_files,
    _start_running_marker,
)


def test_sweep_detects_missing_example_index(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"sample_id": "a", "metadata": {}}) + "\n", encoding="utf-8")

    assert _jsonl_has_row_aligned_example_index(path) is False


def test_sweep_accepts_row_aligned_example_index(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"sample_id": "a", "example_index": 0, "metadata": {"example_index": 0}},
        {"sample_id": "b", "example_index": 1, "metadata": {"example_index": 1}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert _jsonl_has_row_aligned_example_index(path) is True


def test_sweep_materializes_missing_generated_files(tmp_path: Path) -> None:
    base = {
        "paths": {"data_dir": "data", "prepared_dir": "data/prepared"},
        "training": {"dataset_path": "data/prepared/train.jsonl"},
        "memory": {},
    }
    output_root = tmp_path / "sweep"
    variant = {
        "config_path": str(output_root / "configs" / "trial.yaml"),
        "adapter_path": str(output_root / "adapters" / "trial.safetensors"),
        "gate_dir": str(output_root / "gates" / "trial" / "quick"),
        "run_id": "sweep_trial",
        "params": {
            "rank": 16,
            "learning_rate": 4e-6,
            "dropout": 0.03,
            "max_grad_norm": 0.5,
            "weight_decay": 0.01,
            "lora_num_layers": 28,
            "iters": 800,
            "save_interval": 400,
            "source_root": str(tmp_path),
            "output_root": str(output_root),
            "runs_dir": str(output_root / "runs"),
        },
    }

    _materialize_variant_files(base, variant)

    config_path = Path(variant["config_path"])
    assert config_path.exists()
    assert Path(variant["adapter_path"]).parent.exists()
    assert Path(variant["gate_dir"]).exists()
    assert (output_root / "runs" / "sweep_trial").exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["training"]["lora_rank"] == 16
    assert config["training"]["coverage"]["enabled"] is True


def test_sweep_running_marker_refuses_live_pid(tmp_path: Path) -> None:
    output_root = tmp_path / "sweep"
    marker = _start_running_marker(output_root)
    try:
        try:
            _start_running_marker(output_root)
        except RuntimeError as exc:
            assert str(os.getpid()) in str(exc)
        else:
            raise AssertionError("expected live marker to refuse a second sweep")
    finally:
        _finish_running_marker(marker, 1)

    assert not marker.exists()
    assert (output_root / ".FINISHED.json").exists()
