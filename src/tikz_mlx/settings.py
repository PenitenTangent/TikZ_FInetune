from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: yaml.Loader, node: yaml.Node, deep: bool = False) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = getattr(key_node, "start_mark", None)
            location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
            raise ValueError(f"Duplicate YAML key{location}: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _resolve_path(root_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path


def _resolve_optional_path(root_dir: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(root_dir, value)


@dataclass(slots=True)
class PathsConfig:
    root_dir: Path
    data_dir: Path
    prepared_dir: Path
    manifests_dir: Path
    outputs_dir: Path
    runs_dir: Path
    cache_dir: Path

    @classmethod
    def from_mapping(cls, root_dir: Path, mapping: dict[str, Any]) -> "PathsConfig":
        return cls(
            root_dir=root_dir,
            data_dir=root_dir / mapping["data_dir"],
            prepared_dir=root_dir / mapping["prepared_dir"],
            manifests_dir=root_dir / mapping["manifests_dir"],
            outputs_dir=root_dir / mapping["outputs_dir"],
            runs_dir=root_dir / mapping["runs_dir"],
            cache_dir=root_dir / mapping["cache_dir"],
        )


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    max_context_tokens: int
    max_output_tokens: int
    image_resize_shape: tuple[int, int]
    temperature: float
    top_p: float
    top_k: int
    enable_thinking: bool

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ModelConfig":
        shape = mapping["image_resize_shape"]
        max_context_tokens = int(mapping["max_context_tokens"])
        max_output_tokens = int(mapping["max_output_tokens"])
        image_resize_shape = (int(shape[0]), int(shape[1]))
        temperature = float(mapping["temperature"])
        top_p = float(mapping["top_p"])
        top_k = int(mapping["top_k"])
        if max_context_tokens <= 0:
            raise ValueError("model.max_context_tokens must be positive.")
        if max_output_tokens <= 0:
            raise ValueError("model.max_output_tokens must be positive.")
        if image_resize_shape[0] <= 0 or image_resize_shape[1] <= 0:
            raise ValueError("model.image_resize_shape values must be positive.")
        if temperature < 0.0:
            raise ValueError("model.temperature must be non-negative.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("model.top_p must be in the range (0, 1].")
        if top_k <= 0:
            raise ValueError("model.top_k must be positive.")
        return cls(
            model_id=mapping["model_id"],
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            image_resize_shape=image_resize_shape,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=bool(mapping.get("enable_thinking", False)),
        )


@dataclass(slots=True)
class CompilerConfig:
    tectonic_binary: Path
    timeout_seconds: int
    keep_logs: bool
    keep_intermediates: bool
    untrusted: bool

    @classmethod
    def from_mapping(cls, root_dir: Path, mapping: dict[str, Any]) -> "CompilerConfig":
        binary = Path(mapping["tectonic_binary"]).expanduser()
        if not binary.is_absolute():
            binary = (root_dir / binary).resolve()
        timeout_seconds = int(mapping["timeout_seconds"])
        if timeout_seconds <= 0:
            raise ValueError("compiler.timeout_seconds must be positive.")
        return cls(
            tectonic_binary=binary,
            timeout_seconds=timeout_seconds,
            keep_logs=bool(mapping["keep_logs"]),
            keep_intermediates=bool(mapping["keep_intermediates"]),
            untrusted=bool(mapping["untrusted"]),
        )


@dataclass(slots=True)
class MemoryConfig:
    batch_size: int
    gradient_checkpointing: bool
    gradient_accumulation_steps: int
    freeze_vision: bool
    clear_cache_between_retries: bool
    retry_cache_policy: str
    peak_memory_abort_gb: float
    wired_limit_fraction: float

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "MemoryConfig":
        batch_size = int(mapping["batch_size"])
        gradient_accumulation_steps = int(mapping["gradient_accumulation_steps"])
        clear_cache_between_retries = bool(mapping["clear_cache_between_retries"])
        retry_cache_policy = str(
            mapping.get(
                "retry_cache_policy",
                "clear" if clear_cache_between_retries else "none",
            )
        ).lower()
        peak_memory_abort_gb = float(mapping["peak_memory_abort_gb"])
        wired_limit_fraction = float(mapping["wired_limit_fraction"])
        if batch_size <= 0:
            raise ValueError("memory.batch_size must be positive.")
        if gradient_accumulation_steps <= 0:
            raise ValueError("memory.gradient_accumulation_steps must be positive.")
        if retry_cache_policy not in {"none", "clear", "reload"}:
            raise ValueError("memory.retry_cache_policy must be one of: none, clear, reload.")
        if peak_memory_abort_gb <= 0.0:
            raise ValueError("memory.peak_memory_abort_gb must be positive.")
        if not 0.0 < wired_limit_fraction < 1.0:
            raise ValueError("memory.wired_limit_fraction must be in the range (0, 1).")
        return cls(
            batch_size=batch_size,
            gradient_checkpointing=bool(mapping["gradient_checkpointing"]),
            gradient_accumulation_steps=gradient_accumulation_steps,
            freeze_vision=bool(mapping["freeze_vision"]),
            clear_cache_between_retries=clear_cache_between_retries,
            retry_cache_policy=retry_cache_policy,
            peak_memory_abort_gb=peak_memory_abort_gb,
            wired_limit_fraction=wired_limit_fraction,
        )


@dataclass(slots=True)
class DatasetConfig:
    min_chars: int
    max_chars: int
    split_seed: int
    supported_environments: tuple[str, ...]
    reject_external_dependencies: bool
    deduplicate: bool
    drop_truncated_records: bool

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "DatasetConfig":
        min_chars = int(mapping["min_chars"])
        max_chars = int(mapping["max_chars"])
        split_seed = int(mapping.get("split_seed", 17))
        supported_environments = tuple(mapping["supported_environments"])
        if min_chars <= 0:
            raise ValueError("dataset.min_chars must be positive.")
        if max_chars < min_chars:
            raise ValueError("dataset.max_chars must be greater than or equal to dataset.min_chars.")
        if split_seed < 0:
            raise ValueError("dataset.split_seed must be zero or greater.")
        if not supported_environments:
            raise ValueError("dataset.supported_environments must not be empty.")
        return cls(
            min_chars=min_chars,
            max_chars=max_chars,
            split_seed=split_seed,
            supported_environments=supported_environments,
            reject_external_dependencies=bool(mapping["reject_external_dependencies"]),
            deduplicate=bool(mapping["deduplicate"]),
            drop_truncated_records=bool(mapping.get("drop_truncated_records", False)),
        )


@dataclass(slots=True)
class DecodingConfig:
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None
    no_repeat_ngram_size: int | None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "DecodingConfig":
        def _optional_int(key: str) -> int | None:
            value = mapping.get(key)
            if value in (None, ""):
                return None
            return int(value)

        def _optional_float(key: str) -> float | None:
            value = mapping.get(key)
            if value in (None, ""):
                return None
            return float(value)

        max_tokens = _optional_int("max_tokens")
        temperature = _optional_float("temperature")
        top_p = _optional_float("top_p")
        top_k = _optional_int("top_k")
        min_p = _optional_float("min_p")
        repetition_penalty = _optional_float("repetition_penalty")
        no_repeat_ngram_size = _optional_int("no_repeat_ngram_size")

        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("inference decoding max_tokens must be positive when provided.")
        if temperature is not None and temperature < 0.0:
            raise ValueError("inference decoding temperature must be non-negative when provided.")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("inference decoding top_p must be in the range (0, 1] when provided.")
        if top_k is not None and top_k <= 0:
            raise ValueError("inference decoding top_k must be positive when provided.")
        if min_p is not None and not 0.0 < min_p <= 1.0:
            raise ValueError("inference decoding min_p must be in the range (0, 1] when provided.")
        if repetition_penalty is not None and repetition_penalty <= 0.0:
            raise ValueError("inference decoding repetition_penalty must be positive when provided.")
        if no_repeat_ngram_size is not None and no_repeat_ngram_size <= 0:
            raise ValueError("inference decoding no_repeat_ngram_size must be positive when provided.")

        return cls(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )


@dataclass(slots=True)
class InferenceConfig:
    max_retries: int
    repair_max_retries: int
    visual_refine_on_success: bool
    debug_grid_step_px: int
    initial_candidates: int
    compile_repair_candidates: int
    visual_repair_candidates: int
    initial_decoding: DecodingConfig
    compile_repair_decoding: DecodingConfig
    visual_repair_decoding: DecodingConfig

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "InferenceConfig":
        max_retries = int(mapping["max_retries"])
        repair_max_retries = int(mapping["repair_max_retries"])
        debug_grid_step_px = int(mapping["debug_grid_step_px"])
        initial_candidates = int(mapping.get("initial_candidates", 1))
        compile_repair_candidates = int(mapping.get("compile_repair_candidates", 1))
        visual_repair_candidates = int(mapping.get("visual_repair_candidates", 1))
        if max_retries < 0:
            raise ValueError("inference.max_retries must be zero or greater.")
        if repair_max_retries < 0:
            raise ValueError("inference.repair_max_retries must be zero or greater.")
        if repair_max_retries > max_retries:
            raise ValueError("inference.repair_max_retries cannot exceed inference.max_retries.")
        if debug_grid_step_px <= 0:
            raise ValueError("inference.debug_grid_step_px must be positive.")
        if initial_candidates <= 0:
            raise ValueError("inference.initial_candidates must be positive.")
        if compile_repair_candidates <= 0:
            raise ValueError("inference.compile_repair_candidates must be positive.")
        if visual_repair_candidates <= 0:
            raise ValueError("inference.visual_repair_candidates must be positive.")
        return cls(
            max_retries=max_retries,
            repair_max_retries=repair_max_retries,
            visual_refine_on_success=bool(mapping["visual_refine_on_success"]),
            debug_grid_step_px=debug_grid_step_px,
            initial_candidates=initial_candidates,
            compile_repair_candidates=compile_repair_candidates,
            visual_repair_candidates=visual_repair_candidates,
            initial_decoding=DecodingConfig.from_mapping(mapping.get("initial_decoding") or {}),
            compile_repair_decoding=DecodingConfig.from_mapping(mapping.get("compile_repair_decoding") or {}),
            visual_repair_decoding=DecodingConfig.from_mapping(mapping.get("visual_repair_decoding") or {}),
        )


@dataclass(slots=True)
class Stage2TrainingConfig:
    enabled: bool
    allow_full_training: bool
    dataset_path: Path
    val_dataset_path: Path | None
    gold_eval_dataset_path: Path | None
    output_path: Path
    checkpoint_dir: Path
    reward_cache_dir: Path
    telemetry_path: Path
    reward_model_id: str | None
    resume_adapter_path: Path | None
    reward_backend: str
    rollout_group_size: int
    smoke_rollout_group_size: int
    learning_rate: float
    max_rollout_tokens: int
    smoke_max_rollout_tokens: int
    temperature: float
    top_p: float
    top_k: int
    clip_epsilon_low: float
    clip_epsilon_high: float
    beta: float
    normalize_by_max_length: bool
    scale_advantages_by_std: bool
    mask_truncated_completions: bool
    steps_per_report: int
    steps_per_save: int
    iters: int
    smoke_steps: int
    cache_reference_artifacts: bool
    reward_compile_floor: float
    reward_format_floor: float
    promotion_min_reward: float
    promotion_min_compile_rate: float
    fail_on_invalid_rollout: bool
    dead_signal_watchdog_enabled: bool
    dead_signal_watchdog_windows: int
    dead_signal_watchdog_min_format_reject_rate: float
    dead_signal_watchdog_min_truncated_rate: float
    rollout_debug_save_every: int
    rollout_debug_max_dirs: int
    rollout_debug_force_final_save: bool

    @classmethod
    def from_mapping(cls, root_dir: Path, mapping: dict[str, Any]) -> "Stage2TrainingConfig":
        rollout_group_size = int(mapping.get("rollout_group_size", 8))
        smoke_rollout_group_size = int(mapping.get("smoke_rollout_group_size", 2))
        learning_rate = float(mapping.get("learning_rate", 1e-5))
        max_rollout_tokens = int(mapping.get("max_rollout_tokens", 2048))
        smoke_max_rollout_tokens = int(mapping.get("smoke_max_rollout_tokens", 256))
        temperature = float(mapping.get("temperature", 1.0))
        top_p = float(mapping.get("top_p", 0.9))
        top_k = int(mapping.get("top_k", 64))
        clip_epsilon_low = float(mapping.get("clip_epsilon_low", 0.2))
        clip_epsilon_high = float(mapping.get("clip_epsilon_high", 0.28))
        beta = float(mapping.get("beta", 0.0))
        steps_per_report = int(mapping.get("steps_per_report", 1))
        steps_per_save = int(mapping.get("steps_per_save", 1))
        iters = int(mapping.get("iters", 1))
        smoke_steps = int(mapping.get("smoke_steps", 1))
        reward_backend = str(mapping.get("reward_backend", "emd")).lower()
        reward_compile_floor = float(mapping.get("reward_compile_floor", 0.05))
        reward_format_floor = float(mapping.get("reward_format_floor", 0.01))
        promotion_min_reward = float(mapping.get("promotion_min_reward", 0.0))
        promotion_min_compile_rate = float(mapping.get("promotion_min_compile_rate", 0.0))
        dead_signal_watchdog_windows = int(mapping.get("dead_signal_watchdog_windows", 3))
        dead_signal_watchdog_min_format_reject_rate = float(
            mapping.get("dead_signal_watchdog_min_format_reject_rate", 0.95)
        )
        dead_signal_watchdog_min_truncated_rate = float(
            mapping.get("dead_signal_watchdog_min_truncated_rate", 0.95)
        )
        rollout_debug_save_every = int(mapping.get("rollout_debug_save_every", 1))
        rollout_debug_max_dirs = int(mapping.get("rollout_debug_max_dirs", 0))
        rollout_debug_force_final_save = bool(mapping.get("rollout_debug_force_final_save", True))

        if rollout_debug_save_every <= 0:
            raise ValueError("training.stage2.rollout_debug_save_every must be positive.")
        if rollout_debug_max_dirs < 0:
            raise ValueError("training.stage2.rollout_debug_max_dirs must be non-negative.")

        if rollout_group_size <= 0:
            raise ValueError("training.stage2.rollout_group_size must be positive.")
        if smoke_rollout_group_size <= 0:
            raise ValueError("training.stage2.smoke_rollout_group_size must be positive.")
        if smoke_rollout_group_size > rollout_group_size:
            raise ValueError(
                "training.stage2.smoke_rollout_group_size cannot exceed training.stage2.rollout_group_size."
            )
        if learning_rate <= 0.0:
            raise ValueError("training.stage2.learning_rate must be positive.")
        if max_rollout_tokens <= 0:
            raise ValueError("training.stage2.max_rollout_tokens must be positive.")
        if smoke_max_rollout_tokens <= 0:
            raise ValueError("training.stage2.smoke_max_rollout_tokens must be positive.")
        if smoke_max_rollout_tokens > max_rollout_tokens:
            raise ValueError(
                "training.stage2.smoke_max_rollout_tokens cannot exceed training.stage2.max_rollout_tokens."
            )
        if temperature < 0.0:
            raise ValueError("training.stage2.temperature must be non-negative.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("training.stage2.top_p must be in the range (0, 1].")
        if top_k <= 0:
            raise ValueError("training.stage2.top_k must be positive.")
        if clip_epsilon_low < 0.0:
            raise ValueError("training.stage2.clip_epsilon_low must be non-negative.")
        if clip_epsilon_high < 0.0:
            raise ValueError("training.stage2.clip_epsilon_high must be non-negative.")
        if beta < 0.0:
            raise ValueError("training.stage2.beta must be non-negative.")
        if steps_per_report <= 0:
            raise ValueError("training.stage2.steps_per_report must be positive.")
        if steps_per_save <= 0:
            raise ValueError("training.stage2.steps_per_save must be positive.")
        if iters <= 0:
            raise ValueError("training.stage2.iters must be positive.")
        if smoke_steps <= 0:
            raise ValueError("training.stage2.smoke_steps must be positive.")
        if reward_backend not in {"emd", "selfsim", "gemini"}:
            raise ValueError("training.stage2.reward_backend must be either 'emd', 'selfsim' or 'gemini'.")
        if reward_compile_floor < 0.0:
            raise ValueError("training.stage2.reward_compile_floor must be non-negative.")
        if reward_format_floor < 0.0:
            raise ValueError("training.stage2.reward_format_floor must be non-negative.")
        if reward_format_floor > reward_compile_floor:
            raise ValueError(
                "training.stage2.reward_format_floor cannot exceed training.stage2.reward_compile_floor."
            )
        if promotion_min_reward < 0.0:
            raise ValueError("training.stage2.promotion_min_reward must be non-negative.")
        if not 0.0 <= promotion_min_compile_rate <= 1.0:
            raise ValueError("training.stage2.promotion_min_compile_rate must be in the range [0, 1].")
        if dead_signal_watchdog_windows <= 0:
            raise ValueError("training.stage2.dead_signal_watchdog_windows must be positive.")
        if not 0.0 <= dead_signal_watchdog_min_format_reject_rate <= 1.0:
            raise ValueError(
                "training.stage2.dead_signal_watchdog_min_format_reject_rate must be in the range [0, 1]."
            )
        if not 0.0 <= dead_signal_watchdog_min_truncated_rate <= 1.0:
            raise ValueError(
                "training.stage2.dead_signal_watchdog_min_truncated_rate must be in the range [0, 1]."
            )

        dataset_value = mapping.get("dataset_path", "data/prepared/train_stage2.jsonl")
        val_dataset_value = mapping.get("val_dataset_path", "data/prepared/val_stage2.jsonl")
        gold_eval_dataset_value = mapping.get("gold_eval_dataset_path", "data/prepared/gold_eval_stage2.jsonl")
        output_value = mapping.get("output_path", "runs/tikz_stage2_adapter.safetensors")
        checkpoint_value = mapping.get("checkpoint_dir", "runs/stage2_checkpoints")
        reward_cache_value = mapping.get("reward_cache_dir", ".cache/stage2/references")
        telemetry_value = mapping.get("telemetry_path", "runs/stage2_checkpoints/metrics.jsonl")

        return cls(
            enabled=bool(mapping.get("enabled", False)),
            allow_full_training=bool(mapping.get("allow_full_training", False)),
            dataset_path=_resolve_path(root_dir, dataset_value),
            val_dataset_path=_resolve_optional_path(root_dir, val_dataset_value),
            gold_eval_dataset_path=_resolve_optional_path(root_dir, gold_eval_dataset_value),
            output_path=_resolve_path(root_dir, output_value),
            checkpoint_dir=_resolve_path(root_dir, checkpoint_value),
            reward_cache_dir=_resolve_path(root_dir, reward_cache_value),
            telemetry_path=_resolve_path(root_dir, telemetry_value),
            reward_model_id=mapping.get("reward_model_id"),
            resume_adapter_path=_resolve_optional_path(root_dir, mapping.get("resume_adapter_path")),
            reward_backend=reward_backend,
            rollout_group_size=rollout_group_size,
            smoke_rollout_group_size=smoke_rollout_group_size,
            learning_rate=learning_rate,
            max_rollout_tokens=max_rollout_tokens,
            smoke_max_rollout_tokens=smoke_max_rollout_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            clip_epsilon_low=clip_epsilon_low,
            clip_epsilon_high=clip_epsilon_high,
            beta=beta,
            normalize_by_max_length=bool(mapping.get("normalize_by_max_length", True)),
            scale_advantages_by_std=bool(mapping.get("scale_advantages_by_std", False)),
            mask_truncated_completions=bool(mapping.get("mask_truncated_completions", True)),
            steps_per_report=steps_per_report,
            steps_per_save=steps_per_save,
            iters=iters,
            smoke_steps=smoke_steps,
            cache_reference_artifacts=bool(mapping.get("cache_reference_artifacts", True)),
            reward_compile_floor=reward_compile_floor,
            reward_format_floor=reward_format_floor,
            promotion_min_reward=promotion_min_reward,
            promotion_min_compile_rate=promotion_min_compile_rate,
            fail_on_invalid_rollout=bool(mapping.get("fail_on_invalid_rollout", True)),
            dead_signal_watchdog_enabled=bool(mapping.get("dead_signal_watchdog_enabled", True)),
            dead_signal_watchdog_windows=dead_signal_watchdog_windows,
            dead_signal_watchdog_min_format_reject_rate=dead_signal_watchdog_min_format_reject_rate,
            dead_signal_watchdog_min_truncated_rate=dead_signal_watchdog_min_truncated_rate,
            rollout_debug_save_every=rollout_debug_save_every,
            rollout_debug_max_dirs=rollout_debug_max_dirs,
            rollout_debug_force_final_save=rollout_debug_force_final_save,
        )


@dataclass(slots=True)
class CoverageConfig:
    enabled: bool
    strict_fingerprint: bool
    require_state_for_resume: bool
    save_interval_steps: int
    order_mode: str
    order_seed_base: int
    state_file_name: str
    lock_file_name: str
    epoch_orders_dir_name: str

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "CoverageConfig":
        order_mode = str(mapping.get("order_mode", "epoch-shuffle")).lower()
        save_interval_steps = int(mapping.get("save_interval_steps", 10))
        order_seed_base = int(mapping.get("order_seed_base", 17))
        state_file_name = str(mapping.get("state_file_name", "coverage_state.json"))
        lock_file_name = str(mapping.get("lock_file_name", "train.lock"))
        epoch_orders_dir_name = str(mapping.get("epoch_orders_dir_name", "epoch_orders"))

        if save_interval_steps <= 0:
            raise ValueError("training.coverage.save_interval_steps must be positive.")
        if order_mode not in {"epoch-shuffle", "fixed"}:
            raise ValueError("training.coverage.order_mode must be one of: epoch-shuffle, fixed.")
        if order_seed_base < 0:
            raise ValueError("training.coverage.order_seed_base must be zero or greater.")
        if not state_file_name:
            raise ValueError("training.coverage.state_file_name must not be empty.")
        if not lock_file_name:
            raise ValueError("training.coverage.lock_file_name must not be empty.")
        if not epoch_orders_dir_name:
            raise ValueError("training.coverage.epoch_orders_dir_name must not be empty.")

        return cls(
            enabled=bool(mapping.get("enabled", False)),
            strict_fingerprint=bool(mapping.get("strict_fingerprint", True)),
            require_state_for_resume=bool(mapping.get("require_state_for_resume", True)),
            save_interval_steps=save_interval_steps,
            order_mode=order_mode,
            order_seed_base=order_seed_base,
            state_file_name=state_file_name,
            lock_file_name=lock_file_name,
            epoch_orders_dir_name=epoch_orders_dir_name,
        )


@dataclass(slots=True)
class CollapseProbeConfig:
    enabled: bool
    interval_steps: int
    max_failures: int
    save_checkpoint_on_pass: bool
    start_step: int
    probe_at_end_only: bool
    allowed_failures: int

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "CollapseProbeConfig":
        interval_steps = int(mapping.get("interval_steps", 500))
        max_failures = int(mapping.get("max_failures", 1))
        start_step = int(mapping.get("start_step", 0))
        probe_at_end_only = bool(mapping.get("probe_at_end_only", False))
        allowed_failures = int(mapping.get("allowed_failures", 0))
        if interval_steps <= 0:
            raise ValueError("training.collapse_probe.interval_steps must be positive.")
        if max_failures <= 0:
            raise ValueError("training.collapse_probe.max_failures must be positive.")
        if start_step < 0:
            raise ValueError("training.collapse_probe.start_step must be non-negative.")
        if allowed_failures < 0:
            raise ValueError("training.collapse_probe.allowed_failures must be non-negative.")
        return cls(
            enabled=bool(mapping.get("enabled", True)),
            interval_steps=interval_steps,
            max_failures=max_failures,
            save_checkpoint_on_pass=bool(mapping.get("save_checkpoint_on_pass", True)),
            start_step=start_step,
            probe_at_end_only=probe_at_end_only,
            allowed_failures=allowed_failures,
        )


@dataclass(slots=True)
class TrainingConfig:
    train_dataset_path: Path
    pretokenized_cache_path: Path | None
    pretokenized_packed_cache_path: Path | None  # packed ids (.npy), requires _boundaries.npy sibling
    val_dataset_path: Path | None
    gold_eval_dataset_path: Path | None
    require_nonempty_validation_dataset: bool
    require_nonempty_gold_eval_dataset: bool
    min_validation_compilation_rate: float
    validation_compile_probe_limit: int
    dry_run_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float | None
    epochs: int
    steps_per_save: int
    checkpoint_keep_last: int
    checkpoint_pin_iterations: tuple[int, ...]
    checkpoint_cleanup_interval_seconds: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_num_layers: int | None
    steps_per_eval: int
    val_batches: int
    val_metadata_length_warn_fraction_of_context: float
    auto_resume_latest_checkpoint: bool
    train_on_completions: bool
    completion_mask_preflight_enabled: bool
    completion_mask_preflight_rows: int
    completion_mask_preflight_min_marker_hit_rate: float
    completion_mask_preflight_min_mask_zero_fraction: float
    reward_weighted_loss: bool
    reward_weight_field: str
    reward_weight_floor: float
    reward_weight_ceil: float
    reward_weight_path: Path | None
    syntax_weighted_loss: bool
    syntax_weight_path: Path | None
    syntax_structural_weight: float
    syntax_command_weight: float
    syntax_coordinate_weight: float
    repetition_unlikelihood_enabled: bool
    repetition_unlikelihood_weight: float
    repetition_unlikelihood_window: int
    repetition_unlikelihood_min_context: int
    repetition_unlikelihood_warmup_steps: int
    max_seq_length_schedule: tuple[tuple[float, int], ...]
    allow_full_training: bool
    resume_adapter_path: Path | None
    assistant_id: int | None
    coverage: CoverageConfig
    collapse_probe: CollapseProbeConfig
    stage2: Stage2TrainingConfig
    static_critic_training_gate: bool  # plan §2.5: drop training records with critical static violations
    static_critic_max_violations: int  # max tolerated violations before a record is dropped
    repair_before_training: bool  # plan §2.3: compile-and-repair each JSONL completion before SFT
    repair_before_training_timeout: float  # per-sample tectonic timeout seconds during the pre-flight repair
    lr_warmup_fraction: float  # fraction of total steps used for LR warmup (default 0.10 for fresh runs)
    learning_verification_min_updates: int
    learning_verification_min_loss_delta: float

    @classmethod
    def from_mapping(cls, root_dir: Path, mapping: dict[str, Any]) -> "TrainingConfig":
        train_dataset_path = _resolve_path(root_dir, mapping.get("dataset_path", "data/prepared/train.jsonl"))
        pretokenized_cache_path = _resolve_optional_path(root_dir, mapping.get("pretokenized_cache_path"))
        pretokenized_packed_cache_path = _resolve_optional_path(root_dir, mapping.get("pretokenized_packed_cache_path"))
        val_dataset_path = _resolve_optional_path(root_dir, mapping.get("val_dataset_path", "data/prepared/val.jsonl"))
        gold_eval_dataset_path = _resolve_optional_path(
            root_dir,
            mapping.get("gold_eval_dataset_path", "data/prepared/gold_eval.jsonl"),
        )
        require_nonempty_validation_dataset = bool(mapping.get("require_nonempty_validation_dataset", True))
        require_nonempty_gold_eval_dataset = bool(mapping.get("require_nonempty_gold_eval_dataset", True))
        min_validation_compilation_rate = float(mapping.get("min_validation_compilation_rate", 0.0))
        validation_compile_probe_limit = int(mapping.get("validation_compile_probe_limit", 25))
        dry_run_steps = int(mapping["dry_run_steps"])
        learning_rate = float(mapping["learning_rate"])
        weight_decay = float(mapping.get("weight_decay", 0.01))  # default per plan recommendation
        max_grad_norm_raw = mapping.get("max_grad_norm")
        max_grad_norm = float(max_grad_norm_raw) if max_grad_norm_raw is not None else None
        epochs = int(mapping["epochs"])
        steps_per_save = int(mapping.get("steps_per_save", 100))
        checkpoint_keep_last = int(mapping.get("checkpoint_keep_last", 0))
        pin_raw = mapping.get("checkpoint_pin_iterations")
        if pin_raw in (None, []):
            checkpoint_pin_iterations: tuple[int, ...] = ()
        else:
            if not isinstance(pin_raw, (list, tuple)):
                raise ValueError("training.checkpoint_pin_iterations must be a list of integers.")
            try:
                checkpoint_pin_iterations = tuple(sorted({int(x) for x in pin_raw}))
            except (TypeError, ValueError) as exc:
                raise ValueError("training.checkpoint_pin_iterations must be a list of integers.") from exc
            for pin in checkpoint_pin_iterations:
                if pin <= 0:
                    raise ValueError("training.checkpoint_pin_iterations entries must be positive.")
        checkpoint_cleanup_interval_seconds = int(mapping.get("checkpoint_cleanup_interval_seconds", 30))
        lora_rank = int(mapping["lora_rank"])
        lora_alpha = int(mapping["lora_alpha"])
        lora_dropout = float(mapping["lora_dropout"])
        lora_num_layers_raw = mapping.get("lora_num_layers")
        lora_num_layers = int(lora_num_layers_raw) if lora_num_layers_raw is not None else None
        steps_per_eval = int(mapping.get("steps_per_eval", 200))
        val_batches = int(mapping.get("val_batches", 25))
        val_metadata_length_warn_fraction_of_context = float(
            mapping.get("val_metadata_length_warn_fraction_of_context", 0.5)
        )
        completion_mask_preflight_enabled = bool(mapping.get("completion_mask_preflight_enabled", True))
        completion_mask_preflight_rows = int(mapping.get("completion_mask_preflight_rows", 256))
        completion_mask_preflight_min_marker_hit_rate = float(
            mapping.get("completion_mask_preflight_min_marker_hit_rate", 0.9)
        )
        completion_mask_preflight_min_mask_zero_fraction = float(
            mapping.get("completion_mask_preflight_min_mask_zero_fraction", 0.01)
        )
        reward_weighted_loss = bool(mapping.get("reward_weighted_loss", False))
        reward_weight_field = str(mapping.get("reward_weight_field", "sample_weight"))
        reward_weight_floor = float(mapping.get("reward_weight_floor", 0.0))
        reward_weight_ceil = float(mapping.get("reward_weight_ceil", 1.0))
        reward_weight_path = _resolve_optional_path(root_dir, mapping.get("reward_weight_path"))
        syntax_weighted_loss = bool(mapping.get("syntax_weighted_loss", False))
        syntax_weight_path = _resolve_optional_path(root_dir, mapping.get("syntax_weight_path"))
        syntax_structural_weight = float(mapping.get("syntax_structural_weight", 5.0))
        syntax_command_weight = float(mapping.get("syntax_command_weight", 2.0))
        syntax_coordinate_weight = float(mapping.get("syntax_coordinate_weight", 1.0))
        repetition_unlikelihood_enabled = bool(mapping.get("repetition_unlikelihood_enabled", False))
        repetition_unlikelihood_weight = float(mapping.get("repetition_unlikelihood_weight", 0.05))
        repetition_unlikelihood_window = int(mapping.get("repetition_unlikelihood_window", 64))
        repetition_unlikelihood_min_context = int(mapping.get("repetition_unlikelihood_min_context", 16))

        repetition_unlikelihood_warmup_steps = int(mapping.get("repetition_unlikelihood_warmup_steps", 0))

        seq_schedule_raw = mapping.get("max_seq_length_schedule")
        schedule_points: list[tuple[float, int]] = []
        if seq_schedule_raw not in (None, ""):
            if not isinstance(seq_schedule_raw, (list, tuple)):
                raise ValueError("training.max_seq_length_schedule must be a list.")
            for entry in seq_schedule_raw:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    fraction = float(entry[0])
                    max_len = int(entry[1])
                elif isinstance(entry, dict):
                    fraction = float(entry.get("fraction"))
                    max_len = int(entry.get("max_seq_length"))
                else:
                    raise ValueError(
                        "training.max_seq_length_schedule entries must be [fraction, max_seq_length] pairs "
                        "or mappings with keys {fraction, max_seq_length}."
                    )
                if not 0.0 <= fraction <= 1.0:
                    raise ValueError("training.max_seq_length_schedule fractions must be in [0, 1].")
                if max_len <= 0:
                    raise ValueError("training.max_seq_length_schedule max_seq_length must be positive.")
                schedule_points.append((fraction, max_len))
            schedule_points.sort(key=lambda item: item[0])
            for i in range(1, len(schedule_points)):
                if schedule_points[i][0] <= schedule_points[i - 1][0]:
                    raise ValueError("training.max_seq_length_schedule fractions must be strictly increasing.")
        max_seq_length_schedule = tuple(schedule_points)
        if dry_run_steps <= 0:
            raise ValueError("training.dry_run_steps must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive.")
        if weight_decay < 0.0:
            raise ValueError("training.weight_decay must be non-negative.")
        static_critic_training_gate = bool(mapping.get("static_critic_training_gate", False))
        static_critic_max_violations = int(mapping.get("static_critic_max_violations", 0))
        repair_before_training = bool(mapping.get("repair_before_training", False))
        repair_before_training_timeout = float(mapping.get("repair_before_training_timeout", 10.0))
        lr_warmup_fraction = float(mapping.get("lr_warmup_fraction", 0.10))
        learning_verification_min_updates = int(mapping.get("learning_verification_min_updates", 5))
        learning_verification_min_loss_delta = float(mapping.get("learning_verification_min_loss_delta", 0.001))
        if not 0.0 <= lr_warmup_fraction < 1.0:
            raise ValueError("training.lr_warmup_fraction must be in the range [0, 1).")
        if learning_verification_min_updates < 0:
            raise ValueError("training.learning_verification_min_updates must be non-negative.")
        if learning_verification_min_loss_delta < 0.0:
            raise ValueError("training.learning_verification_min_loss_delta must be non-negative.")
        if epochs <= 0:
            raise ValueError("training.epochs must be positive.")
        if steps_per_save <= 0:
            raise ValueError("training.steps_per_save must be positive.")
        if checkpoint_keep_last < 0:
            raise ValueError("training.checkpoint_keep_last must be zero or greater.")
        if checkpoint_cleanup_interval_seconds <= 0:
            raise ValueError("training.checkpoint_cleanup_interval_seconds must be positive.")
        if lora_rank <= 0:
            raise ValueError("training.lora_rank must be positive.")
        if lora_alpha <= 0:
            raise ValueError("training.lora_alpha must be positive.")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError("training.lora_dropout must be in the range [0, 1).")
        if not 0.0 <= min_validation_compilation_rate <= 1.0:
            raise ValueError("training.min_validation_compilation_rate must be in the range [0, 1].")
        if validation_compile_probe_limit <= 0:
            raise ValueError("training.validation_compile_probe_limit must be positive.")
        if completion_mask_preflight_rows <= 0:
            raise ValueError("training.completion_mask_preflight_rows must be positive.")
        if not 0.0 <= completion_mask_preflight_min_marker_hit_rate <= 1.0:
            raise ValueError(
                "training.completion_mask_preflight_min_marker_hit_rate must be in the range [0, 1]."
            )
        if not 0.0 <= completion_mask_preflight_min_mask_zero_fraction <= 1.0:
            raise ValueError(
                "training.completion_mask_preflight_min_mask_zero_fraction must be in the range [0, 1]."
            )
        if val_batches <= 0:
            raise ValueError("training.val_batches must be positive.")
        if not 0.0 < val_metadata_length_warn_fraction_of_context <= 1.0:
            raise ValueError(
                "training.val_metadata_length_warn_fraction_of_context must be in the range (0, 1]."
            )
        if not reward_weight_field:
            raise ValueError("training.reward_weight_field must not be empty.")
        if reward_weight_floor < 0.0:
            raise ValueError("training.reward_weight_floor must be non-negative.")
        if reward_weight_ceil < reward_weight_floor:
            raise ValueError("training.reward_weight_ceil must be >= training.reward_weight_floor.")
        if syntax_structural_weight <= 0.0:
            raise ValueError("training.syntax_structural_weight must be positive.")
        if syntax_command_weight <= 0.0:
            raise ValueError("training.syntax_command_weight must be positive.")
        if syntax_coordinate_weight <= 0.0:
            raise ValueError("training.syntax_coordinate_weight must be positive.")
        if repetition_unlikelihood_weight < 0.0:
            raise ValueError("training.repetition_unlikelihood_weight must be non-negative.")
        if repetition_unlikelihood_window <= 0:
            raise ValueError("training.repetition_unlikelihood_window must be positive.")
        if repetition_unlikelihood_min_context < 0:
            raise ValueError("training.repetition_unlikelihood_min_context must be non-negative.")
        if repetition_unlikelihood_warmup_steps < 0:
            raise ValueError("training.repetition_unlikelihood_warmup_steps must be non-negative.")
        return cls(
            train_dataset_path=train_dataset_path,
            pretokenized_cache_path=pretokenized_cache_path,
            pretokenized_packed_cache_path=pretokenized_packed_cache_path,
            val_dataset_path=val_dataset_path,
            gold_eval_dataset_path=gold_eval_dataset_path,
            require_nonempty_validation_dataset=require_nonempty_validation_dataset,
            require_nonempty_gold_eval_dataset=require_nonempty_gold_eval_dataset,
            min_validation_compilation_rate=min_validation_compilation_rate,
            validation_compile_probe_limit=validation_compile_probe_limit,
            dry_run_steps=dry_run_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            steps_per_save=steps_per_save,
            checkpoint_keep_last=checkpoint_keep_last,
            checkpoint_pin_iterations=checkpoint_pin_iterations,
            checkpoint_cleanup_interval_seconds=checkpoint_cleanup_interval_seconds,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_num_layers=lora_num_layers,
            steps_per_eval=steps_per_eval,
            val_batches=val_batches,
            val_metadata_length_warn_fraction_of_context=val_metadata_length_warn_fraction_of_context,
            auto_resume_latest_checkpoint=bool(mapping.get("auto_resume_latest_checkpoint", False)),
            train_on_completions=bool(mapping["train_on_completions"]),
            completion_mask_preflight_enabled=completion_mask_preflight_enabled,
            completion_mask_preflight_rows=completion_mask_preflight_rows,
            completion_mask_preflight_min_marker_hit_rate=completion_mask_preflight_min_marker_hit_rate,
            completion_mask_preflight_min_mask_zero_fraction=completion_mask_preflight_min_mask_zero_fraction,
            reward_weighted_loss=reward_weighted_loss,
            reward_weight_field=reward_weight_field,
            reward_weight_floor=reward_weight_floor,
            reward_weight_ceil=reward_weight_ceil,
            reward_weight_path=reward_weight_path,
            syntax_weighted_loss=syntax_weighted_loss,
            syntax_weight_path=syntax_weight_path,
            syntax_structural_weight=syntax_structural_weight,
            syntax_command_weight=syntax_command_weight,
            syntax_coordinate_weight=syntax_coordinate_weight,
            repetition_unlikelihood_enabled=repetition_unlikelihood_enabled,
            repetition_unlikelihood_weight=repetition_unlikelihood_weight,
            repetition_unlikelihood_window=repetition_unlikelihood_window,
            repetition_unlikelihood_min_context=repetition_unlikelihood_min_context,
            repetition_unlikelihood_warmup_steps=repetition_unlikelihood_warmup_steps,
            max_seq_length_schedule=max_seq_length_schedule,
            allow_full_training=bool(mapping["allow_full_training"]),
            resume_adapter_path=_resolve_optional_path(root_dir, mapping.get("resume_adapter_path")),
            assistant_id=mapping.get("assistant_id"),
            coverage=CoverageConfig.from_mapping(mapping.get("coverage", {})),
            collapse_probe=CollapseProbeConfig.from_mapping(mapping.get("collapse_probe", {})),
            stage2=Stage2TrainingConfig.from_mapping(root_dir, mapping.get("stage2", {})),
            static_critic_training_gate=static_critic_training_gate,
            static_critic_max_violations=static_critic_max_violations,
            repair_before_training=repair_before_training,
            repair_before_training_timeout=repair_before_training_timeout,
            lr_warmup_fraction=lr_warmup_fraction,
            learning_verification_min_updates=learning_verification_min_updates,
            learning_verification_min_loss_delta=learning_verification_min_loss_delta,
        )


@dataclass(slots=True)
class PipelineConfig:
    config_path: Path
    paths: PathsConfig
    model: ModelConfig
    compiler: CompilerConfig
    memory: MemoryConfig
    dataset: DatasetConfig
    inference: InferenceConfig
    training: TrainingConfig


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=UniqueKeyLoader) or {}

    root_dir = path.parent.parent
    return PipelineConfig(
        config_path=path,
        paths=PathsConfig.from_mapping(root_dir, data["paths"]),
        model=ModelConfig.from_mapping(data["model"]),
        compiler=CompilerConfig.from_mapping(root_dir, data["compiler"]),
        memory=MemoryConfig.from_mapping(data["memory"]),
        dataset=DatasetConfig.from_mapping(data["dataset"]),
        inference=InferenceConfig.from_mapping(data["inference"]),
        training=TrainingConfig.from_mapping(root_dir, data["training"]),
    )


def ensure_runtime_directories(config: PipelineConfig) -> None:
    directories = (
        config.paths.data_dir,
        config.paths.prepared_dir,
        config.paths.manifests_dir,
        config.paths.outputs_dir,
        config.paths.runs_dir,
        config.paths.cache_dir,
        config.training.stage2.checkpoint_dir,
        config.training.stage2.reward_cache_dir,
        config.training.train_dataset_path.parent,
        config.training.stage2.dataset_path.parent,
        config.training.stage2.telemetry_path.parent,
    )
    optional_directories = (
        config.training.val_dataset_path.parent if config.training.val_dataset_path is not None else None,
        config.training.gold_eval_dataset_path.parent if config.training.gold_eval_dataset_path is not None else None,
        config.training.stage2.val_dataset_path.parent if config.training.stage2.val_dataset_path is not None else None,
        (
            config.training.stage2.gold_eval_dataset_path.parent
            if config.training.stage2.gold_eval_dataset_path is not None
            else None
        ),
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    for directory in optional_directories:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)


def require_training_opt_in(config: PipelineConfig, dry_run: bool) -> None:
    if dry_run:
        return
    if not config.training.allow_full_training:
        raise RuntimeError(
            "Full training is disabled in config. Set training.allow_full_training=true "
            "only after the build-first acceptance gate passes."
        )


def require_stage2_training_opt_in(config: PipelineConfig, dry_run: bool) -> None:
    if dry_run:
        return
    if not config.training.stage2.allow_full_training:
        raise RuntimeError(
            "Stage 2 training is disabled in config. Set training.stage2.allow_full_training=true "
            "only after the stage-2 smoke gate passes."
        )
