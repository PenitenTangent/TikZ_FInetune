from pathlib import Path

import pytest

from tikz_mlx.checkpointing import checkpoint_metadata_path
from tikz_mlx.mlx_runtime import MlxRuntimeUnavailableError, import_mlx_core
from tikz_mlx.settings import load_config
from tikz_mlx.schemas import Stage2Sample
from tikz_mlx.train_stage2 import (
    Stage2Rollout,
    Stage2RolloutBatch,
    _build_stage2_rollout_prompt,
    _compute_stage2_config_fingerprint,
    _contains_markdown_fence,
    _extract_markdown_fenced_content,
    _prepare_candidate_code_for_reward,
    _extract_stage2_example_indices,
    _prune_stage2_checkpoints,
    _resolve_stage2_telemetry_path,
    _stage2_dead_signal_window_triggered,
    _strict_stage2_resume_requested,
    _drgrpo_loss,
    compute_group_advantages,
    plan_stage2_training,
    shape_stage2_reward,
    should_promote_checkpoint,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lora_prod.yaml"

try:
    mx = import_mlx_core()
    _MLX_SKIP_REASON = ""
except (ImportError, MlxRuntimeUnavailableError) as exc:
    mx = None
    _MLX_SKIP_REASON = str(exc)

REQUIRES_MLX = pytest.mark.skipif(mx is None, reason=_MLX_SKIP_REASON)


class _DummyOutputs:
    def __init__(self, logits):
        self.logits = logits


class _ConstantLogitModel:
    def __call__(self, inputs, pixel_values, attention_mask):
        batch_size, seq_len = inputs.shape
        logits = mx.array([[[0.0, 2.0, 0.0]] * seq_len for _ in range(batch_size)], dtype=mx.float32)
        return _DummyOutputs(logits)


def test_compute_group_advantages_centers_rewards() -> None:
    advantages = compute_group_advantages([0.2, 0.7, 1.1])
    assert advantages == pytest.approx([-0.4666666666666667, 0.033333333333333326, 0.43333333333333335])


def test_compute_group_advantages_can_scale_by_std() -> None:
    advantages = compute_group_advantages([0.0, 1.0], scale_by_std=True)
    assert advantages == [-1.0, 1.0]


def test_plan_stage2_training_uses_smoke_settings_and_resume_path() -> None:
    config = load_config(CONFIG_PATH)
    resume_path = Path("runs/stage2_resume.safetensors").resolve()
    plan = plan_stage2_training(
        config,
        resume_adapter_path=resume_path,
        dry_run=True,
        require_full_opt_in=False,
    )
    assert plan.args.rollout_group_size == config.training.stage2.smoke_rollout_group_size
    assert plan.args.max_rollout_tokens == config.training.stage2.smoke_max_rollout_tokens
    assert plan.args.adapter_path == str(resume_path)


def test_plan_stage2_training_assigns_default_run_id_from_output_stem() -> None:
    config = load_config(CONFIG_PATH)
    plan = plan_stage2_training(
        config,
        output_path=Path("runs/custom_stage2_adapter.safetensors"),
        dry_run=True,
        require_full_opt_in=False,
    )
    assert plan.args.run_id == "custom_stage2_adapter"


def test_plan_stage2_training_scopes_telemetry_path_by_run_id() -> None:
    config = load_config(CONFIG_PATH)
    plan = plan_stage2_training(
        config,
        output_path=Path("runs/custom_stage2_adapter.safetensors"),
        dry_run=True,
        require_full_opt_in=False,
    )
    assert Path(plan.args.telemetry_path).name == "metrics_custom_stage2_adapter.jsonl"


def test_resolve_stage2_telemetry_path_supports_run_id_template(tmp_path: Path) -> None:
    telemetry_template = tmp_path / "metrics_{run_id}.jsonl"
    resolved = _resolve_stage2_telemetry_path(telemetry_template, "canary-1")
    assert resolved == (tmp_path / "metrics_canary-1.jsonl").resolve()


def test_stage2_dead_signal_window_triggered_requires_both_rates() -> None:
    assert _stage2_dead_signal_window_triggered(
        average_format_reject_rate=0.97,
        average_truncated_rate=0.96,
        min_format_reject_rate=0.95,
        min_truncated_rate=0.95,
    )
    assert not _stage2_dead_signal_window_triggered(
        average_format_reject_rate=0.97,
        average_truncated_rate=0.94,
        min_format_reject_rate=0.95,
        min_truncated_rate=0.95,
    )


def test_extract_stage2_example_indices_requires_row_alignment() -> None:
    samples = [
        Stage2Sample(sample_id="a", prompt_text="p", metadata={"example_index": 1}),
        Stage2Sample(sample_id="b", prompt_text="p", metadata={"example_index": 0}),
    ]

    with pytest.raises(ValueError):
        _extract_stage2_example_indices(samples)


def test_contains_markdown_fence_detects_fenced_blocks() -> None:
    assert _contains_markdown_fence("```latex\\n\\documentclass[tikz]{standalone}\\n```")
    assert not _contains_markdown_fence("\\documentclass[tikz]{standalone}\\n\\begin{document}\\n\\end{document}")


def test_extract_markdown_fenced_content_prefers_longest_block() -> None:
    response = (
        "```text\\nshort\\n```\\n"
        "```latex\\n"
        "\\documentclass[tikz]{standalone}\\n"
        "\\begin{document}\\n"
        "\\begin{tikzpicture}\\n"
        "\\draw (0,0) -- (1,1);\\n"
        "\\end{tikzpicture}\\n"
        "\\end{document}\\n"
        "```"
    )
    extracted = _extract_markdown_fenced_content(response)
    assert extracted is not None
    assert "\\documentclass[tikz]{standalone}" in extracted


def test_prepare_candidate_code_for_reward_uses_fenced_latex() -> None:
    response = (
        "```latex\\n"
        "\\documentclass[tikz]{standalone}\\n"
        "\\begin{document}\\n"
        "\\begin{tikzpicture}\\n"
        "\\draw (0,0) -- (1,1);\\n"
        "\\end{tikzpicture}\\n"
        "\\end{document}\\n"
        "```"
    )
    generated_code, had_fence = _prepare_candidate_code_for_reward(response)
    assert had_fence is True
    assert "```" not in generated_code
    assert "\\begin{tikzpicture}" in generated_code


def test_prepare_candidate_code_for_reward_extracts_canonical_document_block() -> None:
    response = (
        "Here is the figure.\n"
        "\\documentclass[tikz]{standalone}\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture}\n"
        "\\draw (0,0) -- (1,1);\n"
        "\\end{tikzpicture}\n"
        "\\end{document}\n"
        "Thanks."
    )
    generated_code, had_fence = _prepare_candidate_code_for_reward(response)
    assert had_fence is False
    assert generated_code.startswith("\\documentclass[tikz]{standalone}")
    assert generated_code.endswith("\\end{document}")


def test_prepare_candidate_code_for_reward_wraps_tikzpicture_fragment() -> None:
    response = "\\begin{tikzpicture}\\n\\draw (0,0) -- (1,1);\\n\\end{tikzpicture}"
    generated_code, had_fence = _prepare_candidate_code_for_reward(response)
    assert had_fence is False
    assert generated_code.startswith("\\documentclass[tikz]{standalone}")
    assert "\\begin{document}" in generated_code
    assert "\\end{document}" in generated_code


def test_build_stage2_rollout_prompt_appends_output_contract() -> None:
    prompt = _build_stage2_rollout_prompt("Draw a red triangle")
    assert prompt.startswith("Draw a red triangle")
    assert "Do not repeat or paraphrase the request" in prompt
    assert "\\documentclass[tikz]{standalone}" in prompt


@REQUIRES_MLX
def test_drgrpo_loss_cancels_centered_advantages() -> None:
    current_logprob = float((mx.array([0.0, 2.0, 0.0]) - mx.logsumexp(mx.array([0.0, 2.0, 0.0])))[1].item())
    batch = Stage2RolloutBatch(
        prompt_input_ids=mx.array([[0]], dtype=mx.int32),
        prompt_attention_mask=mx.array([[1]], dtype=mx.int32),
        rollouts=[
            Stage2Rollout(
                token_ids=[1, 1],
                old_logprobs=[current_logprob, current_logprob],
                response_text="",
                generated_code="",
                reward=0.5,
                truncated=False,
            ),
            Stage2Rollout(
                token_ids=[1, 1],
                old_logprobs=[current_logprob, current_logprob],
                response_text="",
                generated_code="",
                reward=-0.5,
                truncated=False,
            ),
        ],
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        beta=0.0,
        normalize_by_max_length=True,
        mask_truncated_completions=True,
    )
    loss = _drgrpo_loss(_ConstantLogitModel(), batch)
    assert abs(float(loss.item())) < 1e-6


@REQUIRES_MLX
def test_drgrpo_loss_masks_truncated_rollouts() -> None:
    current_logprob = float((mx.array([0.0, 2.0, 0.0]) - mx.logsumexp(mx.array([0.0, 2.0, 0.0])))[1].item())
    batch = Stage2RolloutBatch(
        prompt_input_ids=mx.array([[0]], dtype=mx.int32),
        prompt_attention_mask=mx.array([[1]], dtype=mx.int32),
        rollouts=[
            Stage2Rollout(
                token_ids=[1],
                old_logprobs=[current_logprob],
                response_text="",
                generated_code="",
                reward=1.0,
                truncated=True,
            )
        ],
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        beta=0.0,
        normalize_by_max_length=True,
        mask_truncated_completions=True,
    )
    with pytest.raises(RuntimeError, match="active_rollouts=0"):
        _drgrpo_loss(_ConstantLogitModel(), batch)


@REQUIRES_MLX
def test_drgrpo_loss_allows_zero_when_fail_on_invalid_disabled() -> None:
    current_logprob = float((mx.array([0.0, 2.0, 0.0]) - mx.logsumexp(mx.array([0.0, 2.0, 0.0])))[1].item())
    batch = Stage2RolloutBatch(
        prompt_input_ids=mx.array([[0]], dtype=mx.int32),
        prompt_attention_mask=mx.array([[1]], dtype=mx.int32),
        rollouts=[
            Stage2Rollout(
                token_ids=[1],
                old_logprobs=[current_logprob],
                response_text="",
                generated_code="",
                reward=1.0,
                truncated=True,
            )
        ],
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        beta=0.0,
        normalize_by_max_length=True,
        mask_truncated_completions=True,
        fail_on_invalid_rollout=False,
    )
    loss = _drgrpo_loss(_ConstantLogitModel(), batch)
    assert float(loss.item()) == 0.0


def test_shape_stage2_reward_applies_compile_and_format_floors() -> None:
    canonical_code = "\\documentclass[tikz]{standalone}\n\\begin{document}\n\\end{document}"
    assert shape_stage2_reward(
        0.0,
        compiled=True,
        format_ok=True,
        compile_floor=0.05,
        format_floor=0.01,
        generated_code=canonical_code,
        truncated=False,
    ) == pytest.approx(0.05)
    assert shape_stage2_reward(
        0.0,
        compiled=False,
        format_ok=True,
        compile_floor=0.05,
        format_floor=0.01,
        generated_code=canonical_code,
        truncated=False,
    ) == pytest.approx(0.01)
    assert shape_stage2_reward(
        0.2,
        compiled=False,
        format_ok=False,
        compile_floor=0.05,
        format_floor=0.01,
        generated_code=canonical_code,
        truncated=False,
    ) == pytest.approx(0.2)


def test_shape_stage2_reward_penalizes_truncated_rollouts() -> None:
    canonical_code = "\\documentclass[tikz]{standalone}\n\\begin{document}\n\\end{document}"
    reward = shape_stage2_reward(
        0.2,
        compiled=True,
        format_ok=True,
        compile_floor=0.05,
        format_floor=0.01,
        generated_code=canonical_code,
        truncated=True,
    )
    assert reward == pytest.approx(0.12)


def test_shape_stage2_reward_penalizes_repetition_loops() -> None:
    repeated = "\n".join([r"\thagoras"] * 60)
    reward = shape_stage2_reward(
        0.25,
        compiled=True,
        format_ok=True,
        compile_floor=0.05,
        format_floor=0.01,
        generated_code=repeated,
        truncated=False,
    )
    assert reward < 0.05


def test_should_promote_checkpoint_applies_joint_thresholds() -> None:
    assert should_promote_checkpoint(
        average_reward=0.2,
        average_compile_rate=0.7,
        min_reward=0.1,
        min_compile_rate=0.6,
    )
    assert not should_promote_checkpoint(
        average_reward=0.09,
        average_compile_rate=0.7,
        min_reward=0.1,
        min_compile_rate=0.6,
    )
    assert not should_promote_checkpoint(
        average_reward=0.2,
        average_compile_rate=0.59,
        min_reward=0.1,
        min_compile_rate=0.6,
    )


def test_prune_stage2_checkpoints_keeps_latest_numeric_files(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in (1, 2, 3, 4):
        checkpoint = checkpoint_dir / f"{step:07d}_adapters.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint.with_name(f"{checkpoint.name}.metadata.json").write_text("{}", encoding="utf-8")

    best_checkpoint = checkpoint_dir / "best_adapters.safetensors"
    best_checkpoint.write_bytes(b"best")

    deleted = _prune_stage2_checkpoints(checkpoint_dir, keep_last=2)

    assert [path.name for path in deleted] == [
        "0000001_adapters.safetensors",
        "0000002_adapters.safetensors",
    ]
    assert not (checkpoint_dir / "0000001_adapters.safetensors").exists()
    assert not (checkpoint_dir / "0000002_adapters.safetensors").exists()
    assert (checkpoint_dir / "0000003_adapters.safetensors").exists()
    assert (checkpoint_dir / "0000004_adapters.safetensors").exists()
    assert best_checkpoint.exists()


def test_stage2_config_fingerprint_ignores_config_path(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dataset_path = tmp_path / "stage2.jsonl"
    dataset_path.write_text(
        '{"prompt": "p", "reference_code": "\\\\documentclass[tikz]{standalone}\\n\\\\begin{document}\\n\\\\end{document}", "metadata": {"example_index": 0}}\n',
        encoding="utf-8",
    )

    plan = plan_stage2_training(
        config,
        dataset_path=dataset_path,
        dry_run=True,
        require_full_opt_in=False,
    )
    first = _compute_stage2_config_fingerprint(config, plan)

    config.config_path = tmp_path / "alt_config.yaml"
    second = _compute_stage2_config_fingerprint(config, plan)

    assert first == second


def test_prune_stage2_checkpoints_scoped_to_run_id(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step, run_id in ((1, "run-a"), (2, "run-a"), (3, "run-a"), (4, "run-b")):
        checkpoint = checkpoint_dir / f"{step:07d}_adapters.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_metadata_path(checkpoint).write_text(
            f'{{"run_id": "{run_id}"}}',
            encoding="utf-8",
        )

    deleted = _prune_stage2_checkpoints(checkpoint_dir, keep_last=1, run_id="run-a")

    assert [path.name for path in deleted] == [
        "0000001_adapters.safetensors",
        "0000002_adapters.safetensors",
    ]
    assert not (checkpoint_dir / "0000001_adapters.safetensors").exists()
    assert not (checkpoint_dir / "0000002_adapters.safetensors").exists()
    assert (checkpoint_dir / "0000003_adapters.safetensors").exists()
    assert (checkpoint_dir / "0000004_adapters.safetensors").exists()


def test_strict_stage2_resume_requested_requires_adapter_and_state_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage2_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    assert not _strict_stage2_resume_requested(
        adapter_path=str(tmp_path / "adapter_dir"),
        run_dir=run_dir,
        state_file_name="coverage_state.json",
    )

    (run_dir / "coverage_state.json").write_text("{}", encoding="utf-8")

    assert _strict_stage2_resume_requested(
        adapter_path=str(tmp_path / "adapter_dir"),
        run_dir=run_dir,
        state_file_name="coverage_state.json",
    )


def test_strict_stage2_resume_requested_false_without_adapter(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage2_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "coverage_state.json").write_text("{}", encoding="utf-8")

    assert not _strict_stage2_resume_requested(
        adapter_path=None,
        run_dir=run_dir,
        state_file_name="coverage_state.json",
    )


@REQUIRES_MLX
def test_drgrpo_loss_rejects_invalid_rollout_contract() -> None:
    batch = Stage2RolloutBatch(
        prompt_input_ids=mx.array([[0]], dtype=mx.int32),
        prompt_attention_mask=mx.array([[1]], dtype=mx.int32),
        rollouts=[
            Stage2Rollout(
                token_ids=[1, 1],
                old_logprobs=[0.0],
                response_text="",
                generated_code="",
                reward=1.0,
                truncated=False,
            )
        ],
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        beta=0.0,
        normalize_by_max_length=True,
        mask_truncated_completions=True,
        fail_on_invalid_rollout=True,
    )
    with pytest.raises(RuntimeError, match="Rollout contract violated"):
        _drgrpo_loss(_ConstantLogitModel(), batch)
