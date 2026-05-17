from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency
    tqdm = None

from .adapter_config_io import (
    adapter_lora_input_dim_rewrites,
    load_source_lora_hyperparams,
    materialize_lora_handoff_adapter,
    validate_resumed_adapter_lora_hyperparams,
    validate_resumed_adapter_model_and_shape,
)
from .checkpointing import (
    CheckpointContext,
    NamedCheckpointPolicyManager,
    checkpoint_metadata_path,
    extract_iteration_from_checkpoint_name,
    extract_layer_index,
    unwrap_lora_layer,
    resolve_adapter_weights_path,
    utc_now_iso8601,
)
from .dataset import compute_dataset_fingerprint, load_stage2_samples, validate_row_aligned_example_indices
from .mlx_runtime import import_mlx_core, import_mlx_nn
from .model_io import clear_mlx_cache, configure_wired_limit, prepare_adapter_for_mlx_vlm
from .prompting import CANONICAL_TIKZ_DOCUMENT_TEMPLATE, build_gemma_messages, extract_latex_from_response
from .reward.encoder_detikzify import FrozenDetikzifyEncoder
from .reward.pipeline import Stage2RewardPipeline, build_reward_backend
from .schemas import Stage2Sample
from .settings import (
    PipelineConfig,
    ensure_runtime_directories,
    require_stage2_training_opt_in,
)
from .train import (
    EXPECTED_LORA_TARGET_SUFFIXES,
    StrictCoverageTracker,
    _acquire_run_lock,
    _model_hidden_size,
    _release_run_lock,
    _resolve_run_id,
    is_allowed_lora_target_name,
)


TRUNCATION_REWARD_PENALTY = 0.08
REPEATED_LINE_PENALTY_CAP = 0.35
DOMINANT_COMMAND_PENALTY_CAP = 0.20
TOTAL_REPETITION_PENALTY_CAP = 0.45
DOMINANT_COMMAND_RE = re.compile(r"\\[A-Za-z@]+")
MARKDOWN_FENCE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?[ \t]*\n?(.*?)```", re.DOTALL)
CANONICAL_TIKZ_DOCUMENT_BLOCK_RE = re.compile(
    r"(\\documentclass\s*\[tikz\]\s*\{standalone\}.*?\\begin\{document\}.*?\\end\{document\})",
    re.IGNORECASE | re.DOTALL,
)
TIKZPICTURE_BLOCK_RE = re.compile(
    r"(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})",
    re.IGNORECASE | re.DOTALL,
)
STAGE2_ROLLOUT_OUTPUT_CONTRACT = (
    "\n\nReturn only LaTeX code. Do not repeat or paraphrase the request.\n"
    "Do not include Markdown fences, commentary, or analysis text.\n"
    "Use a canonical standalone wrapper and choose a drawing environment that matches the request:\n"
    f"{CANONICAL_TIKZ_DOCUMENT_TEMPLATE}"
)


def _should_persist_rollout_artifacts(
    *,
    global_step: int,
    local_iteration: int,
    total_iters: int,
    save_every: int,
    force_final_save: bool,
) -> bool:
    if save_every <= 1:
        return True
    if force_final_save and local_iteration == total_iters:
        return True
    return global_step % save_every == 0


def _update_rollout_retention_queue(
    queue: dict[Path, int],
    *,
    output_root: Path,
    global_step: int,
    max_kept: int,
) -> list[Path]:
    """Track persisted rollout dirs with recency and return stale dirs to prune."""
    if max_kept <= 0:
        return []

    # Refresh recency for repeated paths.
    queue.pop(output_root, None)
    queue[output_root] = global_step

    stale: list[Path] = []
    while len(queue) > max_kept:
        oldest_path, _oldest_step = next(iter(queue.items()))
        if oldest_path == output_root:
            break
        queue.pop(oldest_path, None)
        stale.append(oldest_path)
    return stale


@dataclass(slots=True)
class Stage2TrainingPlan:
    dataset_path: Path
    output_path: Path
    checkpoint_dir: Path
    reward_cache_dir: Path
    dry_run: bool
    args: argparse.Namespace
    warnings: list[str]


@dataclass(slots=True)
class Stage2Rollout:
    token_ids: list[int]
    old_logprobs: list[float]
    response_text: str
    generated_code: str
    reward: float
    truncated: bool
    compiled: bool = False
    format_ok: bool = False


@dataclass(slots=True)
class Stage2RolloutBatch:
    prompt_input_ids: Any
    prompt_attention_mask: Any
    rollouts: list[Stage2Rollout]
    clip_epsilon_low: float
    clip_epsilon_high: float
    beta: float
    normalize_by_max_length: bool
    mask_truncated_completions: bool
    fail_on_invalid_rollout: bool = True


@dataclass(slots=True)
class Stage2RolloutDiagnostics:
    rollout_count: int = 0
    fence_hits: int = 0
    format_rejects: int = 0
    compile_fails: int = 0
    truncated: int = 0


def compute_group_advantages(rewards: list[float], scale_by_std: bool = False) -> list[float]:
    """Compute group-relative advantages for DR-GRPO.

    Args:
        rewards: List of rewards for a group of rollouts from the same prompt.
        scale_by_std: If True, normalize advantages by the group standard deviation.

    Returns:
        List of advantages (reward - group_mean).
    """
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    advantages = [reward - mean_reward for reward in rewards]
    if not scale_by_std:
        return advantages

    variance = sum(value * value for value in advantages) / len(advantages)
    std = math.sqrt(variance)
    if std <= 1e-8:
        return advantages
    return [value / std for value in advantages]


def _dominant_failure_reason(*, fence_hits: int, format_rejects: int, compile_fails: int, truncated: int) -> str:
    counts = {
        "fence_hit": fence_hits,
        "format_reject": format_rejects,
        "compile_fail": compile_fails,
        "truncated": truncated,
    }
    reason, count = max(counts.items(), key=lambda item: item[1])
    if count <= 0:
        return "none"
    return reason


def _stage2_dead_signal_window_triggered(
    *,
    average_format_reject_rate: float,
    average_truncated_rate: float,
    min_format_reject_rate: float,
    min_truncated_rate: float,
) -> bool:
    return (
        average_format_reject_rate >= min_format_reject_rate
        and average_truncated_rate >= min_truncated_rate
    )


def shape_stage2_reward(
    raw_reward: float,
    *,
    compiled: bool,
    format_ok: bool,
    compile_floor: float,
    format_floor: float,
    generated_code: str,
    truncated: bool,
) -> float:
    """Apply floors and penalties to the raw visual reward.

    Args:
        raw_reward: The base visual similarity score (e.g., EMD).
        compiled: Whether the code compiled successfully.
        format_ok: Whether the code contains a valid TikZ environment.
        compile_floor: The minimum reward for a successful compilation.
        format_floor: The minimum reward for a valid environment format.
        generated_code: The raw generated LaTeX string (used for repetition penalty).
        truncated: Whether the generation was truncated.

    Returns:
        The final shaped reward.
    """
    bounded_reward = max(0.0, float(raw_reward))
    shaped_reward = bounded_reward
    if compiled:
        shaped_reward = max(bounded_reward, compile_floor)
    elif format_ok:
        shaped_reward = max(bounded_reward, format_floor)

    penalty = _repetition_penalty(generated_code)
    if truncated:
        penalty += TRUNCATION_REWARD_PENALTY
    return max(0.0, shaped_reward - penalty)


def _repetition_penalty(generated_code: str) -> float:
    """Calculate a penalty for repetitive line patterns or dominant command loops.

    Args:
        generated_code: The raw generated LaTeX code.

    Returns:
        A float penalty to be subtracted from the reward.
    """
    stripped = generated_code.strip()
    if not stripped:
        return 0.0

    non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    line_penalty = 0.0
    if non_empty_lines:
        most_common_line_count = Counter(non_empty_lines).most_common(1)[0][1]
        if most_common_line_count >= 3:
            line_penalty = min(
                REPEATED_LINE_PENALTY_CAP,
                0.03 * float(most_common_line_count - 2),
            )

    command_penalty = 0.0
    commands = DOMINANT_COMMAND_RE.findall(stripped)
    if len(commands) >= 20:
        dominant_command_count = Counter(commands).most_common(1)[0][1]
        dominant_ratio = float(dominant_command_count) / float(len(commands))
        if dominant_ratio > 0.30:
            command_penalty = min(
                DOMINANT_COMMAND_PENALTY_CAP,
                (dominant_ratio - 0.30) * 0.75,
            )

    return min(TOTAL_REPETITION_PENALTY_CAP, line_penalty + command_penalty)


def should_promote_checkpoint(
    *,
    average_reward: float,
    average_compile_rate: float,
    min_reward: float,
    min_compile_rate: float,
) -> bool:
    """Determine if a checkpoint meets the promotion criteria for Stage 2.

    Args:
        average_reward: Mean reward over the evaluation set.
        average_compile_rate: Mean compilation success rate.
        min_reward: Required promotion threshold for reward.
        min_compile_rate: Required promotion threshold for compile rate.

    Returns:
        True if the checkpoint should be promoted.
    """
    return average_reward >= min_reward and average_compile_rate >= min_compile_rate


def _checkpoint_run_id(checkpoint_path: Path) -> str | None:
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_id = payload.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id
    return None


def _list_stage2_promoted_checkpoints(checkpoint_dir: Path, run_id: str | None = None) -> list[Path]:
    checkpoints: list[Path] = []
    for path in checkpoint_dir.glob("*_adapters.safetensors"):
        if not path.is_file():
            continue
        if extract_iteration_from_checkpoint_name(path) is None:
            continue
        if run_id is not None and _checkpoint_run_id(path) != run_id:
            continue
        checkpoints.append(path)
    checkpoints.sort(
        key=lambda path: (
            extract_iteration_from_checkpoint_name(path) or -1,
            path.name,
        )
    )
    return checkpoints


def _prune_stage2_checkpoints(
    checkpoint_dir: Path,
    keep_last: int,
    run_id: str | None = None,
) -> list[Path]:
    if keep_last <= 0:
        return []

    checkpoints = _list_stage2_promoted_checkpoints(checkpoint_dir, run_id=run_id)
    stale_checkpoints = checkpoints[:-keep_last]
    deleted: list[Path] = []

    for checkpoint in stale_checkpoints:
        try:
            checkpoint.unlink()
            checkpoint_metadata_path(checkpoint).unlink(missing_ok=True)
            deleted.append(checkpoint)
        except FileNotFoundError:
            continue

    return deleted


def _strict_stage2_resume_requested(
    *,
    adapter_path: str | None,
    run_dir: Path,
    state_file_name: str,
) -> bool:
    """Return true only for an actual Stage-2 resume with strict coverage state.

    Stage-2 can bootstrap from a Stage-1 adapter directory on a fresh run.
    That handoff should not require pre-existing Stage-2 coverage state.
    """
    if adapter_path in (None, ""):
        return False
    return (run_dir / state_file_name).exists()


def _resolve_stage2_telemetry_path(base_path: Path, run_id: str) -> Path:
    template = str(base_path)
    if "{run_id}" in template:
        return Path(template.replace("{run_id}", run_id)).expanduser().resolve()

    if base_path.suffix:
        file_name = f"{base_path.stem}_{run_id}{base_path.suffix}"
    else:
        file_name = f"{base_path.name}_{run_id}.jsonl"

    return base_path.with_name(file_name).expanduser().resolve()


def build_stage2_namespace(
    config: PipelineConfig,
    dataset_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    telemetry_path: Path,
    reward_cache_dir: Path,
    dry_run: bool,
    run_id: str,
    resume_adapter_path: Path | None = None,
) -> argparse.Namespace:
    stage2 = config.training.stage2
    return argparse.Namespace(
        model_path=config.model.model_id,
        dataset=str(dataset_path),
        output_path=str(output_path),
        checkpoint_dir=str(checkpoint_dir),
        reward_cache_dir=str(reward_cache_dir),
        reward_model_id=stage2.reward_model_id or config.model.model_id,
        reward_backend=stage2.reward_backend,
        batch_size=config.memory.batch_size,
        iters=stage2.smoke_steps if dry_run else stage2.iters,
        rollout_group_size=stage2.smoke_rollout_group_size if dry_run else stage2.rollout_group_size,
        max_rollout_tokens=stage2.smoke_max_rollout_tokens if dry_run else stage2.max_rollout_tokens,
        learning_rate=stage2.learning_rate,
        steps_per_report=stage2.steps_per_report,
        steps_per_save=stage2.steps_per_save,
        adapter_path=str(resume_adapter_path) if resume_adapter_path is not None else None,
        lora_rank=config.training.lora_rank,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        full_finetune=False,
        train_vision=False,
        grad_checkpoint=config.memory.gradient_checkpointing,
        grad_clip=1.0,
        gradient_accumulation_steps=config.memory.gradient_accumulation_steps,
        temperature=stage2.temperature,
        top_p=stage2.top_p,
        top_k=stage2.top_k,
        clip_epsilon_low=stage2.clip_epsilon_low,
        clip_epsilon_high=stage2.clip_epsilon_high,
        beta=stage2.beta,
        normalize_by_max_length=stage2.normalize_by_max_length,
        scale_advantages_by_std=stage2.scale_advantages_by_std,
        mask_truncated_completions=stage2.mask_truncated_completions,
        cache_reference_artifacts=stage2.cache_reference_artifacts,
        reward_compile_floor=stage2.reward_compile_floor,
        reward_format_floor=stage2.reward_format_floor,
        promotion_min_reward=stage2.promotion_min_reward,
        promotion_min_compile_rate=stage2.promotion_min_compile_rate,
        telemetry_path=str(telemetry_path),
        fail_on_invalid_rollout=stage2.fail_on_invalid_rollout,
        dead_signal_watchdog_enabled=stage2.dead_signal_watchdog_enabled,
        dead_signal_watchdog_windows=stage2.dead_signal_watchdog_windows,
        dead_signal_watchdog_min_format_reject_rate=stage2.dead_signal_watchdog_min_format_reject_rate,
        dead_signal_watchdog_min_truncated_rate=stage2.dead_signal_watchdog_min_truncated_rate,
        lora_num_layers=config.training.lora_num_layers,
        train_mode="drgrpo",
        run_id=run_id,
        rollout_debug_save_every=stage2.rollout_debug_save_every,
        rollout_debug_max_dirs=stage2.rollout_debug_max_dirs,
        rollout_debug_force_final_save=stage2.rollout_debug_force_final_save,
    )


def plan_stage2_training(
    config: PipelineConfig,
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume_adapter_path: str | Path | None = None,
    run_id: str | None = None,
    dry_run: bool = True,
    require_full_opt_in: bool = True,
    iters: int | None = None,
) -> Stage2TrainingPlan:
    """Initialize the training plan, validate paths, and resolve the run ID.

    This function performs pre-flight checks and ensures all runtime directories exist.
    """
    ensure_runtime_directories(config)
    if require_full_opt_in:
        require_stage2_training_opt_in(config, dry_run=dry_run)

    stage2 = config.training.stage2
    dataset = Path(dataset_path) if dataset_path else stage2.dataset_path
    output = Path(output_path) if output_path else stage2.output_path
    checkpoint_dir = stage2.checkpoint_dir
    reward_cache_dir = stage2.reward_cache_dir
    resolved_run_id = _resolve_run_id(output, run_id)
    telemetry_path = _resolve_stage2_telemetry_path(stage2.telemetry_path, resolved_run_id)
    resume_adapter = (
        Path(resume_adapter_path).expanduser().resolve()
        if resume_adapter_path is not None
        else stage2.resume_adapter_path
    )

    warnings: list[str] = []
    if not dataset.exists():
        warnings.append(
            f"Stage 2 dataset path does not exist yet: {dataset}. "
            "RL training cannot start until a stage-2 dataset is prepared."
        )
    if config.memory.batch_size != 1:
        warnings.append("Stage 2 is only validated for batch size 1 on the target Apple Silicon machine.")
    if resume_adapter is None:
        warnings.append(
            "Stage 2 is expected to start from an SFT or prior stage-2 adapter; "
            "no resume adapter path is configured."
        )
    elif not resume_adapter.exists():
        warnings.append(
            f"Stage 2 resume adapter path does not exist yet: {resume_adapter}. "
            "RL training cannot resume from a missing adapter checkpoint."
        )
    if stage2.beta != 0.0:
        warnings.append("Stage 2 beta != 0 is configured, but KL-to-reference is not implemented yet.")

    args = build_stage2_namespace(
        config,
        dataset_path=dataset,
        output_path=output,
        checkpoint_dir=checkpoint_dir,
        telemetry_path=telemetry_path,
        reward_cache_dir=reward_cache_dir,
        dry_run=dry_run,
        run_id=resolved_run_id,
        resume_adapter_path=resume_adapter,
    )
    if iters is not None:
        if iters <= 0:
            raise ValueError("iters override must be positive when provided.")
        args.iters = iters
    return Stage2TrainingPlan(
        dataset_path=dataset,
        output_path=output,
        checkpoint_dir=checkpoint_dir,
        reward_cache_dir=reward_cache_dir,
        dry_run=dry_run,
        args=args,
        warnings=warnings,
    )


def run_stage2_training_smoke_test(
    config: PipelineConfig,
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume_adapter_path: str | Path | None = None,
    run_id: str | None = None,
) -> Stage2TrainingPlan:
    plan = plan_stage2_training(
        config,
        dataset_path=dataset_path,
        output_path=output_path,
        resume_adapter_path=resume_adapter_path,
        run_id=run_id,
        dry_run=True,
        require_full_opt_in=False,
    )
    if not plan.dataset_path.exists():
        raise RuntimeError(f"Stage 2 smoke dataset does not exist: {plan.dataset_path}")
    return _execute_stage2_training(config, plan)


def run_stage2_training(
    config: PipelineConfig,
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume_adapter_path: str | Path | None = None,
    run_id: str | None = None,
    dry_run: bool = True,
    iters: int | None = None,
) -> Stage2TrainingPlan:
    plan = plan_stage2_training(
        config,
        dataset_path=dataset_path,
        output_path=output_path,
        resume_adapter_path=resume_adapter_path,
        run_id=run_id,
        dry_run=dry_run,
        iters=iters,
    )
    if dry_run:
        return plan
    return _execute_stage2_training(config, plan)


def _import_stage2_runtime() -> tuple[Any, ...]:
    try:
        mx = import_mlx_core()
        nn = import_mlx_nn()
        import mlx.optimizers as optim
        from mlx_vlm import load
        from mlx_vlm.generate import generate_step, prepare_inputs
        from mlx_vlm.lora import setup_model_for_training
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.trainer.sft_trainer import grad_checkpoint, save_adapter, tree_map
    except ImportError as exc:
        raise RuntimeError("Stage 2 training requires mlx and mlx-vlm runtime dependencies.") from exc
    return (
        mx,
        nn,
        optim,
        load,
        generate_step,
        prepare_inputs,
        setup_model_for_training,
        apply_chat_template,
        grad_checkpoint,
        save_adapter,
        tree_map,
    )


def _extract_stage2_example_indices(samples: list[Stage2Sample]) -> list[int]:
    example_indices: list[int] = []
    for row_index, sample in enumerate(samples):
        raw_value = sample.metadata.get("example_index")
        if raw_value is None:
            raise RuntimeError(
                "Strict coverage mode requires `example_index` in stage-2 sample metadata. "
                f"Missing at row {row_index}. Re-run dataset splitting before stage-2 training."
            )
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise RuntimeError(
                f"Stage-2 sample at row {row_index} has non-integer `example_index`: {raw_value!r}"
            )
        example_indices.append(int(raw_value))

    validate_row_aligned_example_indices(example_indices)
    return example_indices


def _compute_stage2_config_fingerprint(config: PipelineConfig, plan: Stage2TrainingPlan) -> str:
    payload = {
        "model_id": config.model.model_id,
        "max_context_tokens": config.model.max_context_tokens,
        "dataset_path": str(plan.dataset_path.resolve()),
        "output_path": str(plan.output_path.resolve()),
        "checkpoint_dir": str(Path(plan.args.checkpoint_dir).resolve()),
        "reward_cache_dir": str(Path(plan.args.reward_cache_dir).resolve()),
        "batch_size": plan.args.batch_size,
        "gradient_accumulation_steps": plan.args.gradient_accumulation_steps,
        "learning_rate": plan.args.learning_rate,
        "iters": plan.args.iters,
        "steps_per_save": plan.args.steps_per_save,
        "rollout_group_size": plan.args.rollout_group_size,
        "max_rollout_tokens": plan.args.max_rollout_tokens,
        "clip_epsilon_low": plan.args.clip_epsilon_low,
        "clip_epsilon_high": plan.args.clip_epsilon_high,
        "beta": plan.args.beta,
        "temperature": plan.args.temperature,
        "top_p": plan.args.top_p,
        "top_k": plan.args.top_k,
        "reward_backend": plan.args.reward_backend,
        "reward_compile_floor": plan.args.reward_compile_floor,
        "reward_format_floor": plan.args.reward_format_floor,
        "promotion_min_reward": plan.args.promotion_min_reward,
        "promotion_min_compile_rate": plan.args.promotion_min_compile_rate,
        "dead_signal_watchdog_enabled": plan.args.dead_signal_watchdog_enabled,
        "dead_signal_watchdog_windows": plan.args.dead_signal_watchdog_windows,
        "dead_signal_watchdog_min_format_reject_rate": plan.args.dead_signal_watchdog_min_format_reject_rate,
        "dead_signal_watchdog_min_truncated_rate": plan.args.dead_signal_watchdog_min_truncated_rate,
        "coverage": {
            "order_mode": config.training.coverage.order_mode,
            "order_seed_base": config.training.coverage.order_seed_base,
            "save_interval_steps": config.training.coverage.save_interval_steps,
        },
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_stage2_run_metadata(
    *,
    run_dir: Path,
    run_id: str,
    config: PipelineConfig,
    plan: Stage2TrainingPlan,
    dataset_fingerprint: dict[str, Any],
    config_fingerprint: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_config_path = run_dir / "frozen_config.yaml"
    shutil.copy2(config.config_path, frozen_config_path)

    metadata = {
        "run_id": run_id,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": utc_now_iso8601(),
        "config_path": str(config.config_path),
        "frozen_config_path": str(frozen_config_path),
        "dataset_path": str(plan.dataset_path.resolve()),
        "output_path": str(plan.output_path.resolve()),
        "checkpoint_dir": str(Path(plan.args.checkpoint_dir).resolve()),
        "telemetry_path": str(Path(plan.args.telemetry_path).resolve()),
        "resume_adapter_path": plan.args.adapter_path,
        "dataset_fingerprint": dataset_fingerprint,
        "config_fingerprint": config_fingerprint,
        "named_checkpoint_dir": str((run_dir / "named_checkpoints").resolve()),
    }
    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _execute_stage2_training(config: PipelineConfig, plan: Stage2TrainingPlan) -> Stage2TrainingPlan:
    """The main Stage 2 RL (DR-GRPO) training loop.

    This function handles model loading, adapter shimming, rollout generation,
    reward computation, and policy updates using group-relative advantages.
    """
    if config.memory.batch_size != 1:
        raise RuntimeError("Stage 2 training currently supports batch_size=1 only.")
    if plan.args.beta != 0.0:
        raise RuntimeError("Stage 2 KL-to-reference is not implemented; keep training.stage2.beta at 0.0.")

    (
        mx,
        nn,
        optim,
        load,
        generate_step,
        prepare_inputs,
        setup_model_for_training,
        apply_chat_template,
        grad_checkpoint,
        save_adapter,
        tree_map,
    ) = _import_stage2_runtime()
    import mlx.utils as mx_utils

    samples = load_stage2_samples(plan.dataset_path)
    if not samples:
        raise RuntimeError(f"Stage 2 dataset is empty: {plan.dataset_path}")

    run_id = str(getattr(plan.args, "run_id", _resolve_run_id(plan.output_path, None)))
    run_dir = config.paths.runs_dir / run_id
    dataset_fingerprint_obj = compute_dataset_fingerprint(plan.dataset_path)
    dataset_fingerprint = dataset_fingerprint_obj.to_dict()
    config_fingerprint = _compute_stage2_config_fingerprint(config, plan)

    coverage_tracker: StrictCoverageTracker | None = None
    lock_acquired = False

    if config.training.coverage.enabled:
        _extract_stage2_example_indices(samples)

        resume_requested = _strict_stage2_resume_requested(
            adapter_path=plan.args.adapter_path,
            run_dir=run_dir,
            state_file_name=config.training.coverage.state_file_name,
        )
        if plan.args.adapter_path is not None and not resume_requested:
            plan.warnings.append(
                "Stage-2 coverage state not found for this run; "
                "starting a new strict coverage cursor from the provided adapter."
            )

        coverage_tracker = StrictCoverageTracker(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
            total_examples=len(samples),
            target_steps=plan.args.iters,
            resume_requested=resume_requested,
        )
        _acquire_run_lock(coverage_tracker.lock_path, run_id)
        lock_acquired = True

        remaining_iters = coverage_tracker.remaining_steps
        if remaining_iters <= 0:
            plan.warnings.append(
                "Stage-2 coverage target already reached; no additional iterations were scheduled."
            )
            return plan
        plan.args.iters = remaining_iters
        plan.warnings.append(
            f"Stage-2 strict coverage enabled: epoch={coverage_tracker.state.epoch} "
            f"global_step={coverage_tracker.state.global_step} "
            f"next_example_index={coverage_tracker.state.next_example_index}"
        )

    _write_stage2_run_metadata(
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        plan=plan,
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config_fingerprint,
    )

    named_checkpoint_policy = NamedCheckpointPolicyManager(
        named_dir=run_dir / "named_checkpoints",
        stage="stage2",
        run_id=run_id,
        include_reward_spike=True,
    )

    resume_weights_path = resolve_adapter_weights_path(plan.args.adapter_path)

    def _checkpoint_context(global_step_hint: int | None = None) -> CheckpointContext:
        epoch: int | None = None
        global_step: int | None = None
        batch_in_epoch: int | None = None
        sample_cursor_in_epoch: int | None = None
        epoch_order_checksum: str | None = None

        if coverage_tracker is not None:
            total_examples = max(coverage_tracker.state.total_examples, 1)
            global_step = coverage_tracker.state.global_step
            if global_step_hint is not None and global_step < global_step_hint:
                global_step = global_step_hint
            epoch = global_step // total_examples
            batch_in_epoch = global_step % total_examples
            sample_cursor_in_epoch = batch_in_epoch
            epoch_order_checksum = coverage_tracker.state.epoch_order_checksum
        elif global_step_hint is not None:
            global_step = global_step_hint
            epoch = global_step // max(len(samples), 1)
            batch_in_epoch = global_step % max(len(samples), 1)
            sample_cursor_in_epoch = batch_in_epoch

        return CheckpointContext(
            epoch=epoch,
            global_step=global_step,
            batch_in_epoch=batch_in_epoch,
            sample_cursor_in_epoch=sample_cursor_in_epoch,
            dataset_snapshot_id=dataset_fingerprint_obj.sha256,
            epoch_order_checksum=epoch_order_checksum,
            training_config_fingerprint=config_fingerprint,
        )

    if resume_weights_path is not None:
        resume_context = _checkpoint_context()
        named_checkpoint_policy.record_source_checkpoint(
            checkpoint_path=resume_weights_path,
            checkpoint_role="resume_source",
            context=resume_context,
        )
        named_checkpoint_policy.ensure_policy_init(
            source_checkpoint_path=resume_weights_path,
            context=resume_context,
        )

    reward_encoder = FrozenDetikzifyEncoder(plan.args.reward_model_id, config.memory)
    reward_pipeline = Stage2RewardPipeline(
        config,
        encoder=reward_encoder,
        backend=build_reward_backend(plan.args.reward_backend),
    )

    configure_wired_limit(config.memory)
    validate_resumed_adapter_model_and_shape(
        adapter_path=plan.args.adapter_path,
        expected_model_id=config.model.model_id,
        expected_lora_num_layers=config.training.lora_num_layers,
        expected_target_suffixes=EXPECTED_LORA_TARGET_SUFFIXES,
    )
    model, processor = load(
        plan.args.model_path,
        processor_config={"trust_remote_code": True},
    )
    validate_resumed_adapter_model_and_shape(
        adapter_path=plan.args.adapter_path,
        expected_model_id=config.model.model_id,
        expected_hidden_size=_model_hidden_size(model),
        expected_lora_num_layers=config.training.lora_num_layers,
        expected_target_suffixes=EXPECTED_LORA_TARGET_SUFFIXES,
    )
    handoff_adapter_path: Path | None = None
    if plan.args.adapter_path:
        input_dim_rewrites = adapter_lora_input_dim_rewrites(
            plan.args.adapter_path,
            expected_model_id=config.model.model_id,
        )
        source_hparams = load_source_lora_hyperparams(plan.args.adapter_path) or {}
        sr = source_hparams.get("rank")
        sa = source_hparams.get("alpha")
        tr = config.training.lora_rank
        ta = config.training.lora_alpha
        td = config.training.lora_dropout
        needs_handoff = bool(input_dim_rewrites)
        if sr is not None and tr is not None and sr != tr:
            needs_handoff = True
        elif sa is not None and ta is not None and sa != ta:
            needs_handoff = True

        if needs_handoff:
            handoff_dir = config.paths.runs_dir / "handoffs" / f"stage2_from_{Path(plan.args.adapter_path).stem}_to_rank{tr}"
            materialization = materialize_lora_handoff_adapter(
                source_adapter_path=plan.args.adapter_path,
                target_dir=handoff_dir,
                target_rank=tr,
                target_alpha=ta,
                target_dropout=td,
                input_dim_rewrites=input_dim_rewrites,
            )
            handoff_adapter_path = Path(materialization["adapter_dir"])
            print(f"[Stage2] LoRA handoff adapter materialized at {handoff_adapter_path}", flush=True)
        elif source_hparams:
            validate_resumed_adapter_lora_hyperparams(
                adapter_path=plan.args.adapter_path,
                lora_rank=tr,
                lora_alpha=ta,
                lora_dropout=td,
            )

    prepared_adapter_path = prepare_adapter_for_mlx_vlm(handoff_adapter_path or plan.args.adapter_path)
    model = setup_model_for_training(model, plan.args, prepared_adapter_path)

    # --- LoRA layer limiting ---
    # In Stage 2, we must also enforce LoRA layer limiting to be consistent with Stage 1.
    lora_num_layers = getattr(plan.args, "lora_num_layers", None)
    from mlx_vlm.trainer.lora import LoRaLayer

    layer_nums: set[int] = set()
    for name, _ in model.language_model.named_modules():
        layer_idx = extract_layer_index(name)
        if layer_idx is not None:
            layer_nums.add(layer_idx)
    total_layers = max(layer_nums) + 1 if layer_nums else 0
    cutoff = max(0, total_layers - lora_num_layers) if lora_num_layers is not None else 0

    unwrapped_unexpected: list[str] = []
    for name, module in list(model.language_model.named_modules()):
        if not isinstance(module, LoRaLayer):
            continue
        if not is_allowed_lora_target_name(name, cutoff=cutoff, total_layers=total_layers):
            unwrap_lora_layer(model.language_model, name, module.original_layer)
            unwrapped_unexpected.append(name)

    if unwrapped_unexpected:
        print(
            "[Stage2] Unwrapped unexpected LoRA targets: "
            + ", ".join(unwrapped_unexpected[:50]),
            flush=True,
        )
    if plan.args.grad_checkpoint:
        for module in model.children().values():
            if hasattr(module, "layers"):
                grad_checkpoint(module.layers[0])

    optimizer = optim.Adam(learning_rate=plan.args.learning_rate)
    loss_value_and_grad = nn.value_and_grad(model, _drgrpo_loss)

    report_loss_sum = 0.0
    report_reward_sum = 0.0
    report_compile_sum = 0.0
    report_format_sum = 0.0
    report_rollout_count = 0
    report_fence_hits = 0
    report_format_rejects = 0
    report_compile_fails = 0
    report_truncated = 0
    report_steps = 0

    save_loss_sum = 0.0
    save_reward_sum = 0.0
    save_compile_sum = 0.0
    save_format_sum = 0.0
    save_rollout_count = 0
    save_fence_hits = 0
    save_format_rejects = 0
    save_compile_fails = 0
    save_truncated = 0
    save_steps = 0

    best_reward = float("-inf")
    best_compile_rate = float("-inf")
    best_checkpoint_path: Path | None = None

    checkpoint_dir = Path(plan.args.checkpoint_dir)
    telemetry_path = Path(plan.args.telemetry_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    watchdog_windows_observed = 0
    watchdog_saturated_so_far = True

    model.train()

    progress = None
    iteration_source = range(1, plan.args.iters + 1)
    if tqdm is not None:
        progress = tqdm(iteration_source, total=plan.args.iters, desc="Stage2 training", unit="iter")
        iteration_source = progress

    persisted_rollout_roots: dict[Path, int] = {}

    try:
        for local_iteration in iteration_source:
            if coverage_tracker is not None:
                sample_index = coverage_tracker.peek_next_example_index()
                global_step = coverage_tracker.state.global_step + 1
            else:
                sample_index = (local_iteration - 1) % len(samples)
                global_step = local_iteration

            sample = samples[sample_index]
            rollout_prompt_text = _build_stage2_rollout_prompt(sample.prompt_text)
            prompt = _format_prompt_text(
                apply_chat_template,
                processor,
                model,
                rollout_prompt_text,
                enable_thinking=config.model.enable_thinking,
            )
            save_every = int(getattr(plan.args, "rollout_debug_save_every", 1))
            force_final_save = bool(getattr(plan.args, "rollout_debug_force_final_save", True))
            persist_artifacts = _should_persist_rollout_artifacts(
                global_step=global_step,
                local_iteration=local_iteration,
                total_iters=plan.args.iters,
                save_every=save_every,
                force_final_save=force_final_save,
            )
            if persist_artifacts:
                output_root = config.paths.outputs_dir / "stage2" / sample.sample_id / f"iter_{global_step:07d}"
                output_root.mkdir(parents=True, exist_ok=True)
                (output_root / "prompt.txt").write_text(prompt, encoding="utf-8")
            else:
                output_root = Path(tempfile.mkdtemp(prefix="stage2_rollout_"))
            prompt_inputs = _prepare_prompt_inputs(mx, prepare_inputs, processor, model, prompt)
            rollouts, rollout_diagnostics = _sample_rollout_group(
                mx,
                generate_step,
                model,
                processor,
                prompt_inputs["input_ids"],
                prompt_inputs["attention_mask"],
                sample,
                reward_pipeline,
                output_root=output_root,
                rollout_group_size=plan.args.rollout_group_size,
                max_rollout_tokens=plan.args.max_rollout_tokens,
                temperature=plan.args.temperature,
                top_p=plan.args.top_p,
                top_k=plan.args.top_k,
                min_p=config.inference.initial_decoding.min_p,
                repetition_penalty=config.inference.initial_decoding.repetition_penalty,
                compile_reward_floor=plan.args.reward_compile_floor,
                format_reward_floor=plan.args.reward_format_floor,
            )
            rollout_rewards = [rollout.reward for rollout in rollouts]
            rollout_compile_rate = sum(1 for rollout in rollouts if rollout.compiled) / len(rollouts)
            rollout_format_rate = sum(1 for rollout in rollouts if rollout.format_ok) / len(rollouts)
            advantages = compute_group_advantages(
                rollout_rewards,
                scale_by_std=plan.args.scale_advantages_by_std,
            )
            batch = Stage2RolloutBatch(
                prompt_input_ids=prompt_inputs["input_ids"],
                prompt_attention_mask=prompt_inputs["attention_mask"],
                rollouts=[
                    Stage2Rollout(
                        token_ids=rollout.token_ids,
                        old_logprobs=rollout.old_logprobs,
                        response_text=rollout.response_text,
                        generated_code=rollout.generated_code,
                        reward=advantage,
                        truncated=rollout.truncated,
                        compiled=rollout.compiled,
                        format_ok=rollout.format_ok,
                    )
                    for rollout, advantage in zip(rollouts, advantages, strict=False)
                ],
                clip_epsilon_low=plan.args.clip_epsilon_low,
                clip_epsilon_high=plan.args.clip_epsilon_high,
                beta=plan.args.beta,
                normalize_by_max_length=plan.args.normalize_by_max_length,
                mask_truncated_completions=plan.args.mask_truncated_completions,
                fail_on_invalid_rollout=plan.args.fail_on_invalid_rollout,
            )
            loss, grad = loss_value_and_grad(model, batch)
            # Proper L2 norm-based clipping (same fix as Stage 1)
            flat_grad = mx.concatenate([v.reshape(-1) for _, v in mx_utils.tree_flatten(grad)])
            grad_norm = mx.linalg.norm(flat_grad)
            clip_scale = mx.minimum(plan.args.grad_clip / mx.maximum(grad_norm, 1e-8), 1.0)
            grad = tree_map(lambda g: g * clip_scale, grad)
            optimizer.update(model, grad)
            mx.eval(model.state, optimizer.state, loss)
            clear_mlx_cache()

            if coverage_tracker is not None:
                coverage_tracker.mark_batch_complete(sample_index)

            max_kept = int(getattr(plan.args, "rollout_debug_max_dirs", 0))
            if not persist_artifacts:
                shutil.rmtree(output_root, ignore_errors=True)
            elif max_kept > 0:
                stale_roots = _update_rollout_retention_queue(
                    persisted_rollout_roots,
                    output_root=output_root,
                    global_step=global_step,
                    max_kept=max_kept,
                )
                for old in stale_roots:
                    if old.exists():
                        shutil.rmtree(old, ignore_errors=True)

            loss_value = float(loss.item() if hasattr(loss, "item") else loss)
            reward_value = sum(rollout_rewards) / len(rollout_rewards)
            report_loss_sum += loss_value
            report_reward_sum += reward_value
            report_compile_sum += rollout_compile_rate
            report_format_sum += rollout_format_rate
            report_rollout_count += rollout_diagnostics.rollout_count
            report_fence_hits += rollout_diagnostics.fence_hits
            report_format_rejects += rollout_diagnostics.format_rejects
            report_compile_fails += rollout_diagnostics.compile_fails
            report_truncated += rollout_diagnostics.truncated
            report_steps += 1

            if progress is not None:
                progress.set_postfix(
                    {
                        "reward": f"{reward_value:.4f}",
                        "compile": f"{rollout_compile_rate:.3f}",
                        "format": f"{rollout_format_rate:.3f}",
                    },
                    refresh=False,
                )

            save_loss_sum += loss_value
            save_reward_sum += reward_value
            save_compile_sum += rollout_compile_rate
            save_format_sum += rollout_format_rate
            save_rollout_count += rollout_diagnostics.rollout_count
            save_fence_hits += rollout_diagnostics.fence_hits
            save_format_rejects += rollout_diagnostics.format_rejects
            save_compile_fails += rollout_diagnostics.compile_fails
            save_truncated += rollout_diagnostics.truncated
            save_steps += 1

            if local_iteration % plan.args.steps_per_report == 0 or local_iteration == plan.args.iters:
                denom = max(report_steps, 1)
                rollout_denom = max(report_rollout_count, 1)
                dominant_failure = _dominant_failure_reason(
                    fence_hits=report_fence_hits,
                    format_rejects=report_format_rejects,
                    compile_fails=report_compile_fails,
                    truncated=report_truncated,
                )
                print(
                    f"Iter {global_step}: Stage2 loss {report_loss_sum / denom:.6f}, "
                    f"avg reward {report_reward_sum / denom:.6f}, "
                    f"compile rate {report_compile_sum / denom:.3f}, "
                    f"format rate {report_format_sum / denom:.3f}, "
                    f"fence hits {report_fence_hits}/{rollout_denom}, "
                    f"format rejects {report_format_rejects}/{rollout_denom}, "
                    f"compile fails {report_compile_fails}/{rollout_denom}, "
                    f"truncated {report_truncated}/{rollout_denom}, "
                    f"dominant failure {dominant_failure}",
                    flush=True,
                )
                report_loss_sum = 0.0
                report_reward_sum = 0.0
                report_compile_sum = 0.0
                report_format_sum = 0.0
                report_rollout_count = 0
                report_fence_hits = 0
                report_format_rejects = 0
                report_compile_fails = 0
                report_truncated = 0
                report_steps = 0

            epoch_boundary_reached = False
            if coverage_tracker is not None:
                epoch_boundary_reached = (
                    coverage_tracker.state.batch_cursor_in_epoch == 0 and coverage_tracker.state.global_step > 0
                )
            elif global_step > 0:
                epoch_boundary_reached = global_step % len(samples) == 0

            should_save = (
                local_iteration % plan.args.steps_per_save == 0
                or local_iteration == plan.args.iters
                or epoch_boundary_reached
            )

            if should_save:
                denom = max(save_steps, 1)
                average_save_loss = save_loss_sum / denom
                average_save_reward = save_reward_sum / denom
                average_save_compile_rate = save_compile_sum / denom
                average_save_format_rate = save_format_sum / denom
                save_rollout_denom = max(save_rollout_count, 1)
                average_save_fence_hit_rate = save_fence_hits / save_rollout_denom
                average_save_format_reject_rate = save_format_rejects / save_rollout_denom
                average_save_compile_fail_rate = save_compile_fails / save_rollout_denom
                average_save_truncated_rate = save_truncated / save_rollout_denom
                watchdog_window_triggered = _stage2_dead_signal_window_triggered(
                    average_format_reject_rate=average_save_format_reject_rate,
                    average_truncated_rate=average_save_truncated_rate,
                    min_format_reject_rate=plan.args.dead_signal_watchdog_min_format_reject_rate,
                    min_truncated_rate=plan.args.dead_signal_watchdog_min_truncated_rate,
                )
                if (
                    plan.args.dead_signal_watchdog_enabled
                    and watchdog_windows_observed < plan.args.dead_signal_watchdog_windows
                ):
                    watchdog_windows_observed += 1
                    if not watchdog_window_triggered:
                        watchdog_saturated_so_far = False

                save_context = _checkpoint_context(global_step_hint=global_step)
                save_metrics = {
                    "average_loss": average_save_loss,
                    "average_reward": average_save_reward,
                    "average_compile_rate": average_save_compile_rate,
                    "average_format_rate": average_save_format_rate,
                    "average_fence_hit_rate": average_save_fence_hit_rate,
                    "average_format_reject_rate": average_save_format_reject_rate,
                    "average_compile_fail_rate": average_save_compile_fail_rate,
                    "average_truncated_rate": average_save_truncated_rate,
                }

                save_adapter(model, plan.args.output_path)
                output_checkpoint_path = Path(plan.args.output_path).expanduser().resolve()
                named_checkpoint_policy.record_source_checkpoint(
                    checkpoint_path=output_checkpoint_path,
                    checkpoint_role="adapter_snapshot",
                    context=save_context,
                    metrics=save_metrics,
                )
                if resume_weights_path is None:
                    named_checkpoint_policy.ensure_policy_init(
                        source_checkpoint_path=output_checkpoint_path,
                        context=save_context,
                        metrics=save_metrics,
                    )
                named_checkpoint_policy.update_last(
                    source_checkpoint_path=output_checkpoint_path,
                    context=save_context,
                    metrics=save_metrics,
                )

                promoted = should_promote_checkpoint(
                    average_reward=average_save_reward,
                    average_compile_rate=average_save_compile_rate,
                    min_reward=plan.args.promotion_min_reward,
                    min_compile_rate=plan.args.promotion_min_compile_rate,
                )

                checkpoint: Path | None = None
                if promoted:
                    checkpoint = checkpoint_dir / f"{global_step:07d}_adapters.safetensors"
                    save_adapter(model, checkpoint)
                    named_checkpoint_policy.record_source_checkpoint(
                        checkpoint_path=checkpoint,
                        checkpoint_role="promoted_checkpoint",
                        context=save_context,
                        metrics=save_metrics,
                    )
                    named_checkpoint_policy.update_last(
                        source_checkpoint_path=checkpoint,
                        context=save_context,
                        metrics=save_metrics,
                    )
                    if (
                        average_save_reward > best_reward
                        or (
                            math.isclose(average_save_reward, best_reward)
                            and average_save_compile_rate > best_compile_rate
                        )
                    ):
                        if best_checkpoint_path is not None and best_checkpoint_path.exists():
                            named_checkpoint_policy.update_last_pre_reward_spike(
                                source_checkpoint_path=best_checkpoint_path,
                                context=save_context,
                                metrics=save_metrics,
                            )
                        best_reward = average_save_reward
                        best_compile_rate = average_save_compile_rate
                        best_checkpoint_path = checkpoint_dir / "best_adapters.safetensors"
                        save_adapter(model, best_checkpoint_path)
                        named_checkpoint_policy.record_source_checkpoint(
                            checkpoint_path=best_checkpoint_path,
                            checkpoint_role="best_checkpoint",
                            context=save_context,
                            metrics=save_metrics,
                        )
                        named_checkpoint_policy.update_best_by_eval(
                            source_checkpoint_path=best_checkpoint_path,
                            metric_name="average_reward",
                            metric_value=average_save_reward,
                            higher_is_better=True,
                            context=save_context,
                            metrics=save_metrics,
                        )

                if epoch_boundary_reached:
                    boundary_checkpoint = checkpoint
                    if boundary_checkpoint is None:
                        boundary_checkpoint = checkpoint_dir / f"{global_step:07d}_adapters.safetensors"
                        save_adapter(model, boundary_checkpoint)
                        named_checkpoint_policy.record_source_checkpoint(
                            checkpoint_path=boundary_checkpoint,
                            checkpoint_role="epoch_boundary_checkpoint",
                            context=save_context,
                            metrics=save_metrics,
                        )
                        named_checkpoint_policy.update_last(
                            source_checkpoint_path=boundary_checkpoint,
                            context=save_context,
                            metrics=save_metrics,
                        )
                    named_checkpoint_policy.update_last_epoch_boundary(
                        source_checkpoint_path=boundary_checkpoint,
                        context=save_context,
                        metrics=save_metrics,
                    )

                _append_jsonl_record(
                    telemetry_path,
                    {
                        "run_id": run_id,
                        "iteration": local_iteration,
                        "global_step": global_step,
                        "epoch": save_context.epoch,
                        "batch_in_epoch": save_context.batch_in_epoch,
                        "average_loss": average_save_loss,
                        "average_reward": average_save_reward,
                        "average_compile_rate": average_save_compile_rate,
                        "average_format_rate": average_save_format_rate,
                        "average_fence_hit_rate": average_save_fence_hit_rate,
                        "average_format_reject_rate": average_save_format_reject_rate,
                        "average_compile_fail_rate": average_save_compile_fail_rate,
                        "average_truncated_rate": average_save_truncated_rate,
                        "promoted": promoted,
                        "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
                        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                        "dead_signal_watchdog_window_triggered": watchdog_window_triggered,
                    },
                )

                if checkpoint is not None:
                    print(
                        f"Iter {global_step}: promoted checkpoint {checkpoint} (reward={average_save_reward:.6f}, "
                        f"compile_rate={average_save_compile_rate:.3f}).",
                        flush=True,
                    )
                else:
                    print(
                        f"Iter {global_step}: checkpoint promotion skipped by gate "
                        f"(reward={average_save_reward:.6f}, compile_rate={average_save_compile_rate:.3f}).",
                        flush=True,
                    )

                pruned = _prune_stage2_checkpoints(
                    checkpoint_dir,
                    config.training.checkpoint_keep_last,
                    run_id=run_id,
                )
                if pruned:
                    print(
                        f"Iter {global_step}: pruned {len(pruned)} stale stage-2 checkpoint(s) "
                        f"(keep_last={config.training.checkpoint_keep_last}).",
                        flush=True,
                    )

                if (
                    plan.args.dead_signal_watchdog_enabled
                    and watchdog_windows_observed == plan.args.dead_signal_watchdog_windows
                    and watchdog_saturated_so_far
                ):
                    raise RuntimeError(
                        "Stage-2 dead-signal watchdog triggered: "
                        f"first {plan.args.dead_signal_watchdog_windows} save window(s) exceeded "
                        f"format_reject >= {plan.args.dead_signal_watchdog_min_format_reject_rate:.3f} "
                        f"and truncated >= {plan.args.dead_signal_watchdog_min_truncated_rate:.3f}."
                    )

                save_loss_sum = 0.0
                save_reward_sum = 0.0
                save_compile_sum = 0.0
                save_format_sum = 0.0
                save_rollout_count = 0
                save_fence_hits = 0
                save_format_rejects = 0
                save_compile_fails = 0
                save_truncated = 0
                save_steps = 0
    finally:
        if coverage_tracker is not None:
            coverage_tracker.save(force=True)
        if progress is not None:
            progress.close()
        reward_encoder.unload()
        clear_mlx_cache()
        if lock_acquired and coverage_tracker is not None:
            _release_run_lock(coverage_tracker.lock_path)

    save_adapter(model, plan.args.output_path)
    if best_checkpoint_path is not None:
        print(f"Best promoted stage-2 checkpoint: {best_checkpoint_path}.", flush=True)
    print(f"Saved final stage-2 adapter weights to {plan.args.output_path}.", flush=True)
    return plan


def _append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _format_prompt_text(
    apply_chat_template: Any,
    processor: Any,
    model: Any,
    prompt_text: str,
    *,
    enable_thinking: bool,
) -> str:
    messages = build_gemma_messages(user_text=prompt_text)
    return apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=0,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


def _prepare_prompt_inputs(
    mx: Any,
    prepare_inputs: Any,
    processor: Any,
    model: Any,
    prompt: str,
) -> dict[str, Any]:
    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in {"gemma3", "gemma3n", "gemma4"}
        else True
    )
    inputs = prepare_inputs(
        processor,
        prompts=prompt,
        add_special_tokens=add_special_tokens,
        return_tensors="mlx",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not isinstance(input_ids, mx.array):
        input_ids = mx.array(input_ids)
    if not isinstance(attention_mask, mx.array):
        attention_mask = mx.array(attention_mask)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _sample_rollout_group(
    mx: Any,
    generate_step: Any,
    model: Any,
    processor: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    sample: Stage2Sample,
    reward_pipeline: Stage2RewardPipeline,
    output_root: Path,
    rollout_group_size: int,
    max_rollout_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None,
    repetition_penalty: float | None,
    compile_reward_floor: float,
    format_reward_floor: float,
) -> tuple[list[Stage2Rollout], Stage2RolloutDiagnostics]:
    rollouts: list[Stage2Rollout] = []
    diagnostics = Stage2RolloutDiagnostics()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos_token_ids = _normalize_eos_token_ids(getattr(model.config, "eos_token_id", None))
    if hasattr(tokenizer, "stopping_criteria"):
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

    model.eval()
    for rollout_index in range(rollout_group_size):
        tokens: list[int] = []
        old_logprobs: list[float] = []
        truncated = True
        generator = generate_step(
            prompt_input_ids,
            model,
            None,
            prompt_attention_mask,
            max_tokens=max_rollout_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
        for token, logprobs in generator:
            token_id = int(token)
            tokens.append(token_id)
            old_logprobs.append(float(logprobs[token_id].item()))
            if token_id in eos_token_ids:
                truncated = False
                break
            if hasattr(tokenizer, "stopping_criteria") and tokenizer.stopping_criteria(token_id):
                truncated = False
                break

        response_text = _decode_tokens(tokenizer, tokens)
        generated_code, has_markdown_fence = _prepare_candidate_code_for_reward(response_text)
        is_complete_document = _looks_like_complete_document(generated_code)
        if is_complete_document:
            truncated = False
        rollout_output_dir = output_root / f"rollout_{rollout_index:02d}"
        rollout_output_dir.mkdir(parents=True, exist_ok=True)
        (rollout_output_dir / "response.txt").write_text(response_text, encoding="utf-8")
        (rollout_output_dir / "candidate_for_reward.tex").write_text(generated_code, encoding="utf-8")
        (rollout_output_dir / "rollout_debug.json").write_text(
            json.dumps(
                {
                    "token_count": len(tokens),
                    "token_ids_head": [int(value) for value in tokens[:32]],
                    "has_markdown_fence": has_markdown_fence,
                    "is_complete_document": is_complete_document,
                    "truncated": truncated,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        reward_result = reward_pipeline.score_candidate(
            sample,
            generated_code,
            output_dir=rollout_output_dir,
        )

        diagnostics.rollout_count += 1
        if has_markdown_fence:
            diagnostics.fence_hits += 1
        if not reward_result.format_ok:
            diagnostics.format_rejects += 1
        elif not reward_result.compiled:
            diagnostics.compile_fails += 1
        if truncated:
            diagnostics.truncated += 1

        shaped_reward = shape_stage2_reward(
            reward_result.reward,
            compiled=reward_result.compiled,
            format_ok=reward_result.format_ok,
            compile_floor=compile_reward_floor,
            format_floor=format_reward_floor,
            generated_code=generated_code,
            truncated=truncated,
        )
        rollouts.append(
            Stage2Rollout(
                token_ids=tokens,
                old_logprobs=old_logprobs,
                response_text=response_text,
                generated_code=generated_code,
                reward=shaped_reward,
                truncated=truncated,
                compiled=reward_result.compiled,
                format_ok=reward_result.format_ok,
            )
        )
    model.train()
    return rollouts, diagnostics


def _decode_tokens(tokenizer: Any, tokens: list[int]) -> str:
    if not tokens:
        return ""
    try:
        return str(tokenizer.decode(tokens, skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(tokens))


def _contains_markdown_fence(text: str) -> bool:
    return "```" in text


def _extract_markdown_fenced_content(text: str) -> str | None:
    fenced_segments = [segment.strip() for segment in MARKDOWN_FENCE_BLOCK_RE.findall(text) if segment.strip()]
    if not fenced_segments:
        return None
    # Prefer the largest fenced block in case the response includes small illustrative snippets.
    return max(fenced_segments, key=len)


def _prepare_candidate_code_for_reward(response_text: str) -> tuple[str, bool]:
    has_markdown_fence = _contains_markdown_fence(response_text)
    generated_code = extract_latex_from_response(response_text)

    candidates = [generated_code, response_text]
    fenced_content: str | None = None
    if has_markdown_fence:
        fenced_content = _extract_markdown_fenced_content(response_text)
        if fenced_content is not None:
            fenced_generated_code = extract_latex_from_response(fenced_content)
            if fenced_generated_code.strip():
                candidates.insert(0, fenced_generated_code)
            candidates.append(fenced_content)

    for candidate in candidates:
        canonical = _extract_canonical_tikz_document_block(candidate)
        if canonical is not None:
            return canonical, has_markdown_fence

    for candidate in candidates:
        wrapped = _wrap_tikzpicture_as_document(candidate)
        if wrapped is not None:
            return wrapped, has_markdown_fence

    if generated_code.strip():
        return generated_code, has_markdown_fence
    if fenced_content is not None and fenced_content.strip():
        return fenced_content, has_markdown_fence
    return response_text.strip(), has_markdown_fence


def _extract_canonical_tikz_document_block(text: str) -> str | None:
    match = CANONICAL_TIKZ_DOCUMENT_BLOCK_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _wrap_tikzpicture_as_document(text: str) -> str | None:
    match = TIKZPICTURE_BLOCK_RE.search(text)
    if match is None:
        return None
    tikzpicture = match.group(1).strip()
    return (
        "\\documentclass[tikz]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\begin{document}\n"
        f"{tikzpicture}\n"
        "\\end{document}"
    )


def _looks_like_complete_document(text: str) -> bool:
    lowered = text.lower()
    class_index = lowered.find("\\documentclass")
    begin_index = lowered.find("\\begin{document}")
    end_index = lowered.find("\\end{document}")
    return class_index >= 0 and begin_index > class_index and end_index > begin_index


def _build_stage2_rollout_prompt(prompt_text: str) -> str:
    return f"{prompt_text.strip()}{STAGE2_ROLLOUT_OUTPUT_CONTRACT}"


def _normalize_eos_token_ids(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        values = eos_token_id
    else:
        values = [eos_token_id]
    normalized: set[int] = set()
    for value in values:
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _drgrpo_loss(model: Any, batch: Stage2RolloutBatch) -> Any:
    mx = import_mlx_core()

    max_length = max((len(rollout.token_ids) for rollout in batch.rollouts), default=1)
    objective = mx.array(0.0)
    active_rollouts = 0

    for rollout in batch.rollouts:
        if batch.mask_truncated_completions and rollout.truncated:
            continue
        token_ids = rollout.token_ids
        old_logprob_values = rollout.old_logprobs
        if len(token_ids) != len(old_logprob_values):
            message = (
                "Rollout contract violated: token_ids and old_logprobs must have the same length "
                f"(got {len(token_ids)} vs {len(old_logprob_values)})."
            )
            if batch.fail_on_invalid_rollout:
                raise RuntimeError(message)
            limit = min(len(token_ids), len(old_logprob_values))
            token_ids = token_ids[:limit]
            old_logprob_values = old_logprob_values[:limit]

        if not token_ids:
            continue
        if any(not math.isfinite(value) for value in old_logprob_values):
            message = "Rollout contract violated: old_logprobs must be finite values."
            if batch.fail_on_invalid_rollout:
                raise RuntimeError(message)
            continue

        current_logprobs = _teacher_forced_completion_logprobs(
            mx,
            model,
            batch.prompt_input_ids,
            batch.prompt_attention_mask,
            token_ids,
        )
        old_logprobs = mx.array(old_logprob_values, dtype=current_logprobs.dtype)
        advantage = mx.array(float(rollout.reward), dtype=current_logprobs.dtype)
        ratio = mx.exp(current_logprobs - old_logprobs)
        clipped_ratio = mx.clip(ratio, 1.0 - batch.clip_epsilon_low, 1.0 + batch.clip_epsilon_high)
        surrogate = mx.minimum(ratio * advantage, clipped_ratio * advantage)
        normalizer = max_length if batch.normalize_by_max_length else len(token_ids)
        objective = objective + surrogate.sum() / max(normalizer, 1)
        active_rollouts += 1

    if active_rollouts == 0:
        if batch.fail_on_invalid_rollout:
            raise RuntimeError(
                "All Stage-2 rollouts were masked or invalid (active_rollouts=0). "
                "Disable mask_truncated_completions or increase max_rollout_tokens before retrying."
            )
        return mx.array(0.0)
    return -(objective / active_rollouts)


def _teacher_forced_completion_logprobs(
    mx: Any,
    model: Any,
    prompt_input_ids: Any,
    prompt_attention_mask: Any,
    completion_token_ids: list[int],
) -> Any:
    completion_ids = mx.array([completion_token_ids], dtype=prompt_input_ids.dtype)
    completion_mask = mx.ones_like(completion_ids)
    full_input_ids = mx.concatenate([prompt_input_ids, completion_ids], axis=1)
    full_attention_mask = mx.concatenate([prompt_attention_mask, completion_mask], axis=1)

    inputs = full_input_ids[:, :-1]
    labels = full_input_ids[:, 1:]
    attention_mask = full_attention_mask[:, :-1]
    outputs = model(inputs, None, attention_mask)
    logits = outputs.logits.astype(mx.float32)
    logits = _align_logits_with_labels(mx, logits, labels)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    gathered = mx.take_along_axis(log_probs, mx.expand_dims(labels, axis=-1), axis=-1).squeeze(-1)
    start_index = max(prompt_input_ids.shape[1] - 1, 0)
    return gathered[:, start_index : start_index + len(completion_token_ids)].squeeze(0)


def _align_logits_with_labels(mx: Any, logits: Any, labels: Any) -> Any:
    if logits.shape[1] < labels.shape[1]:
        pad_length = labels.shape[1] - logits.shape[1]
        pad_width = ((0, 0), (0, pad_length), (0, 0))
        return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
    if logits.shape[1] > labels.shape[1]:
        return logits[:, -labels.shape[1] :, :]
    return logits
