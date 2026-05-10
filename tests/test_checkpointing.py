from pathlib import Path

from tikz_mlx.checkpointing import CheckpointContext, NamedCheckpointPolicyManager


def _write_checkpoint(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_named_checkpoint_policy_tracks_last_and_last_prev(tmp_path: Path) -> None:
    manager = NamedCheckpointPolicyManager(
        named_dir=tmp_path / "named",
        stage="stage1",
        run_id="unit-run",
    )
    context = CheckpointContext(global_step=10)

    ckpt_a = _write_checkpoint(tmp_path / "0000010_adapters.safetensors", b"a")
    ckpt_b = _write_checkpoint(tmp_path / "0000020_adapters.safetensors", b"b")

    manager.update_last(source_checkpoint_path=ckpt_a, context=context)
    manager.update_last(source_checkpoint_path=ckpt_b, context=context)

    last_path = tmp_path / "named" / "last.safetensors"
    last_prev_path = tmp_path / "named" / "last_prev.safetensors"

    assert last_path.read_bytes() == b"b"
    assert last_prev_path.read_bytes() == b"a"
    assert (tmp_path / "named" / "last.safetensors.metadata.json").exists()
    assert (tmp_path / "named" / "last_prev.safetensors.metadata.json").exists()


def test_policy_init_remains_immutable_without_override(tmp_path: Path) -> None:
    manager = NamedCheckpointPolicyManager(
        named_dir=tmp_path / "named",
        stage="stage2",
        run_id="unit-run",
        include_reward_spike=True,
    )
    context = CheckpointContext(global_step=1)

    initial = _write_checkpoint(tmp_path / "initial_adapters.safetensors", b"init")
    replacement = _write_checkpoint(tmp_path / "replacement_adapters.safetensors", b"replace")

    manager.ensure_policy_init(source_checkpoint_path=initial, context=context)
    manager.ensure_policy_init(source_checkpoint_path=replacement, context=context)

    policy_init_path = tmp_path / "named" / "policy_init.safetensors"
    assert policy_init_path.read_bytes() == b"init"


def test_best_by_eval_tracks_improvements(tmp_path: Path) -> None:
    manager = NamedCheckpointPolicyManager(
        named_dir=tmp_path / "named",
        stage="stage1",
        run_id="unit-run",
    )
    context = CheckpointContext(global_step=42)

    ckpt_a = _write_checkpoint(tmp_path / "0000042_adapters.safetensors", b"a")
    ckpt_b = _write_checkpoint(tmp_path / "0000043_adapters.safetensors", b"b")

    manager.update_best_by_eval(
        source_checkpoint_path=ckpt_a,
        metric_name="validation_loss",
        metric_value=1.25,
        higher_is_better=False,
        context=context,
    )
    manager.update_best_by_eval(
        source_checkpoint_path=ckpt_b,
        metric_name="validation_loss",
        metric_value=1.50,
        higher_is_better=False,
        context=context,
    )

    best_path = tmp_path / "named" / "best_by_eval.safetensors"
    assert best_path.read_bytes() == b"a"

    manager.update_best_by_eval(
        source_checkpoint_path=ckpt_b,
        metric_name="validation_loss",
        metric_value=0.95,
        higher_is_better=False,
        context=context,
    )
    assert best_path.read_bytes() == b"b"


def test_last_pre_reward_spike_alias_only_when_enabled(tmp_path: Path) -> None:
    manager = NamedCheckpointPolicyManager(
        named_dir=tmp_path / "named",
        stage="stage2",
        run_id="unit-run",
        include_reward_spike=True,
    )
    context = CheckpointContext(global_step=7)
    source = _write_checkpoint(tmp_path / "0000007_adapters.safetensors", b"x")

    result_path = manager.update_last_pre_reward_spike(
        source_checkpoint_path=source,
        context=context,
    )

    assert result_path is not None
    assert result_path.read_bytes() == b"x"
