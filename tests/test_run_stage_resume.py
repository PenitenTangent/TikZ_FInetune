import subprocess
import json
from pathlib import Path

import mlx.core as mx

from tools.run_with_live_progress_tqdm import _parse_iter
from tikz_mlx.checkpointing import checkpoint_metadata_path
from tikz_mlx.train import (
    _maybe_restore_optimizer_state,
    _write_stage_learning_summary,
    load_optimizer_state_sidecar,
    save_optimizer_state_sidecar,
)


def _resume_offset(path: str) -> str:
    result = subprocess.run(
        ["bash", "tools/resume_offset.sh", path],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_resume_offset_strips_leading_zeroes_for_numeric_checkpoints() -> None:
    assert _resume_offset("runs/curriculum_stage1/0001000_adapters.safetensors") == "1000"
    assert _resume_offset("runs/curriculum_stage4/0003000_adapters.safetensors") == "3000"


def test_resume_offset_falls_back_to_zero_for_manual_adapter_names() -> None:
    assert _resume_offset("runs/curriculum_stage4/warmup_adapter.safetensors") == "0"
    assert _resume_offset("runs/curriculum_stage1/warmstart_stage1_3000.safetensors") == "0"


def test_progress_parser_ignores_training_header_but_accepts_global_resume_line() -> None:
    assert _parse_iter("Starting training..., scheduled batches: 14008", 16568) == (None, 16568)
    assert _parse_iter("Resuming from global iteration 2560 / 16568. Running 14008 remaining batches.", 16568) == (
        2560,
        16568,
    )


class _Optimizer:
    def __init__(self, value: float = 1.0) -> None:
        self.state = {"adam": {"m": mx.array([value]), "v": mx.array([value + 1.0])}}


def _write_checkpoint_metadata(
    checkpoint: Path,
    *,
    optimizer_state: Path | None,
    run_id: str = "curriculum_stage0",
    config_fingerprint: str = "cfg",
    dataset_snapshot_id: str = "data",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.08,
    lora_num_layers: int = 42,
) -> None:
    checkpoint.write_bytes(b"adapter")
    checkpoint_metadata_path(checkpoint).write_text(
        json.dumps(
            {
                "global_step": 8,
                "run_id": run_id,
                "training_config_fingerprint": config_fingerprint,
                "dataset_snapshot_id": dataset_snapshot_id,
                "optimizer_state": str(optimizer_state) if optimizer_state is not None else None,
                "resolved_training_config": {
                    "model_id": "mlx-community/gemma-4-e4b-it-6bit",
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "lora_num_layers": lora_num_layers,
                },
            }
        ),
        encoding="utf-8",
    )


def test_optimizer_sidecar_round_trips_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "0000008_adapters.safetensors"
    optimizer = _Optimizer(value=3.0)
    state_path = save_optimizer_state_sidecar(optimizer, checkpoint)
    assert state_path is not None and state_path.exists()

    restored = _Optimizer(value=0.0)
    load_optimizer_state_sidecar(restored, state_path)

    assert restored.state["adam"]["m"].tolist() == [3.0]
    assert restored.state["adam"]["v"].tolist() == [4.0]


def test_optimizer_restore_requires_same_stage_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "0000008_adapters.safetensors"
    source_optimizer = _Optimizer(value=5.0)
    state_path = save_optimizer_state_sidecar(source_optimizer, checkpoint)
    _write_checkpoint_metadata(checkpoint, optimizer_state=state_path)

    target_optimizer = _Optimizer(value=0.0)
    result = _maybe_restore_optimizer_state(
        optimizer=target_optimizer,
        resume_adapter_path=checkpoint,
        run_id="curriculum_stage0",
        config_fingerprint="cfg",
        dataset_snapshot_id="data",
        model_id="mlx-community/gemma-4-e4b-it-6bit",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.08,
        lora_num_layers=42,
    )

    assert result["restored"] is True
    assert target_optimizer.state["adam"]["m"].tolist() == [5.0]


def test_optimizer_restore_resets_on_lora_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "0000008_adapters.safetensors"
    state_path = save_optimizer_state_sidecar(_Optimizer(value=5.0), checkpoint)
    _write_checkpoint_metadata(checkpoint, optimizer_state=state_path, lora_rank=32)

    target_optimizer = _Optimizer(value=0.0)
    result = _maybe_restore_optimizer_state(
        optimizer=target_optimizer,
        resume_adapter_path=checkpoint,
        run_id="curriculum_stage0",
        config_fingerprint="cfg",
        dataset_snapshot_id="data",
        model_id="mlx-community/gemma-4-e4b-it-6bit",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.08,
        lora_num_layers=42,
    )

    assert result["restored"] is False
    assert result["optimizer_state_reset_reason"] == "stage_boundary_or_capacity_change"
    assert "lora_rank" in result["failed_optimizer_restore_checks"]
    assert target_optimizer.state["adam"]["m"].tolist() == [0.0]


def test_stage_learning_summary_warns_on_small_loss_delta(tmp_path: Path) -> None:
    telemetry = tmp_path / "gradient_clip_telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps({"iteration": 8, "train_loss": 1.0, "avg_grad_norm": 2.0, "clipped_step_rate": 0.5, "learning_rate": 1e-6}),
                json.dumps({"iteration": 16, "train_loss": 0.9999, "avg_grad_norm": 1.0, "clipped_step_rate": 0.0, "learning_rate": 9e-7}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _write_stage_learning_summary(
        output_path=tmp_path / "stage_learning_summary.json",
        telemetry_path=telemetry,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        target_optimizer_updates=200,
        min_updates=2,
        min_loss_delta=0.001,
    )

    assert summary["optimizer_update_count"] == 2
    assert summary["learning_warning"]
