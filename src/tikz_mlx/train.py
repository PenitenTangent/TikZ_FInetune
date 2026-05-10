from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
import re
import socket
import shutil
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapter_config_io import validate_resumed_adapter_lora_hyperparams
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
from .compiler import CompilerService
from .curriculum_diagnostics import (
    iter_jsonl_metadata_token_lengths,
    summarize_token_lengths,
    write_phase_boundary_telemetry,
)
from .dataset import (
    build_epoch_example_order,
    compute_dataset_fingerprint,
    compute_example_order_checksum,
    validate_row_aligned_example_indices,
)
from .model_io import clear_mlx_cache, configure_wired_limit, prepare_adapter_for_mlx_vlm
from .mlx_runtime import import_mlx_core, import_mlx_nn
from .quarantine import assert_not_quarantined
from .schemas import CompileStatus
from .settings import PipelineConfig, ensure_runtime_directories, require_training_opt_in
from .sft_train_milestones import train as train_sft_with_milestone_eval
from .static_critic import analyze_tikz_static

DEFAULT_ASSISTANT_ID = 4368
ASSISTANT_MARKER_TEXT_CANDIDATES = (
    "<|turn>model\n",
    "<|turn>assistant\n",
    "<turn|>\n<|turn>model\n",
    "<turn|>\n<|turn>assistant\n",
    "<start_of_turn>model\n",
    "<|start_of_turn|>model\n",
    "<start_of_turn>assistant\n",
    "<|start_of_turn|>assistant\n",
)

CHECKPOINT_METADATA_FILE = "checkpoint_metadata.json"
STRUCTURAL_TOKEN_PATTERN = re.compile(r"^[{}\[\];,\\]+$")
COMMAND_TOKEN_PATTERN = re.compile(r"^\\[A-Za-z@]+$")
COORDINATE_TOKEN_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")


def _read_checkpoint_iteration(adapter_dir: Path) -> int | None:
    """Read the iteration count from checkpoint metadata file."""
    metadata_file = adapter_dir / CHECKPOINT_METADATA_FILE
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file, "r") as f:
            data = json.load(f)
            value = data.get("iteration", data.get("global_step"))
            if value is None:
                return None
            return int(value)
    except (json.JSONDecodeError, OSError):
        return None


def _write_checkpoint_iteration(adapter_dir: Path, iteration: int) -> None:
    """Write the iteration count to checkpoint metadata file."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = adapter_dir / CHECKPOINT_METADATA_FILE
    try:
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "iteration": int(iteration),
                    "global_step": int(iteration),
                    "updated_at": utc_now_iso8601(),
                },
                f,
            )
    except OSError:
        pass  # Non-critical; silently skip if write fails




def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _pack_dataset_sha256(*paths: Path) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def _load_and_validate_pack_audit(
    *,
    packed_path: Path,
    assistant_id: int,
    min_marker_hit_rate: float,
    min_mask_zero_fraction: float,
    reward_weight_path: Path | None = None,
    syntax_weight_path: Path | None = None,
) -> dict[str, Any]:
    """Validate that the packed dataset audit matches the current training config.

    Ensures that the assistant token, marker hit rate, and loss weighting sidecars
    are consistent with the requested training run.
    """
    masks_path = packed_path.with_name(packed_path.stem + "_masks.npy")
    boundaries_path = packed_path.with_name(packed_path.stem + "_boundaries.npy")
    audit_path = packed_path.with_name(packed_path.stem + "_audit.json")
    hash_paths = [packed_path, masks_path, boundaries_path]
    if reward_weight_path is not None:
        hash_paths.append(reward_weight_path)
    if syntax_weight_path is not None:
        hash_paths.append(syntax_weight_path)
    missing = [path for path in hash_paths if not path.exists()]
    if missing:
        missing_str = ", ".join(path.name for path in missing)
        raise RuntimeError(
            "Packed dataset artifacts are missing: "
            f"{missing_str}. Re-run pack_tokenized_dataset.py."
        )
    if not audit_path.exists():
        raise RuntimeError(
            f"No pack audit found for {packed_path.name}. Re-run pack_tokenized_dataset.py."
        )

    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Pack audit is unreadable: {audit_path}") from exc

    live_sha256 = _pack_dataset_sha256(*hash_paths)
    recorded_sha256 = str(payload.get("dataset_sha256", ""))
    if recorded_sha256 != live_sha256:
        raise RuntimeError(
            "Packed dataset audit hash mismatch. "
            f"recorded={recorded_sha256 or '<missing>'}, live={live_sha256}. "
            "Re-run pack_tokenized_dataset.py."
        )

    if bool(payload.get("reward_weighted", False)) != bool(reward_weight_path is not None):
        raise RuntimeError(
            "Packed dataset audit reward-weighting mismatch. "
            "Re-pack the dataset so the audit matches the configured reward_weight_path."
        )
    if bool(payload.get("syntax_weighted", False)) != bool(syntax_weight_path is not None):
        raise RuntimeError(
            "Packed dataset audit syntax-weighting mismatch. "
            "Re-pack the dataset so the audit matches the configured syntax_weight_path."
        )
    if reward_weight_path is None and syntax_weight_path is None:
        metadata_jsonl = payload.get("metadata_jsonl")
        scoring_status = payload.get("scoring_status")
        if metadata_jsonl not in (None, ""):
            raise RuntimeError(
                "Packed dataset audit references metadata_jsonl while training is configured for plain CE. "
                "Re-pack without --metadata-jsonl or enable the matching weighted loss sidecar."
            )
        if scoring_status is not None and scoring_status != "skipped_plain_ce":
            raise RuntimeError(
                "Packed dataset audit scoring_status is incompatible with plain CE training: "
                f"{scoring_status!r}."
            )

    recorded_assistant_id = int(payload.get("assistant_token_used", -1))
    if recorded_assistant_id != assistant_id:
        raise RuntimeError(
            "Packed dataset audit assistant token mismatch. "
            f"recorded={recorded_assistant_id}, configured={assistant_id}. "
            "Re-pack the dataset with the current assistant token."
        )

    marker_hit_rate = float(payload.get("marker_hit_rate", 0.0))
    mask_zero_fraction = float(payload.get("mask_zero_fraction", 0.0))
    if marker_hit_rate < min_marker_hit_rate:
        raise RuntimeError(
            "Packed dataset audit failed marker hit-rate preflight. "
            f"observed={marker_hit_rate:.3f}, required={min_marker_hit_rate:.3f}."
        )
    if mask_zero_fraction < min_mask_zero_fraction:
        raise RuntimeError(
            "Packed dataset audit failed masked-token fraction preflight. "
            f"observed={mask_zero_fraction:.3f}, required={min_mask_zero_fraction:.3f}."
        )

    return payload


def _verify_cache_audit(cache_path: Path, config: PipelineConfig, is_packed: bool = False) -> dict[str, Any]:
    from .prompting import PROMPT_CONTRACT_VERSION, prompt_template_sha256
    
    audit_path = cache_path.with_name(cache_path.stem + "_audit.json")
    if not audit_path.exists():
        if is_packed:
            raise RuntimeError(f"Packed dataset audit is missing: {audit_path}")
        return {}
        
    try:
        with audit_path.open("r", encoding="utf-8") as f:
            audit = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read cache audit {audit_path}: {e}")
    
    # 1. Prompt contract version
    if "prompt_contract_version" in audit:
        if audit["prompt_contract_version"] != PROMPT_CONTRACT_VERSION:
            raise RuntimeError(
                f"Stale pretokenized cache detected! "
                f"Expected contract {PROMPT_CONTRACT_VERSION!r}, got {audit['prompt_contract_version']!r}. "
                "Re-run pretokenization/packing."
            )
    
    # 2. Prompt template sha256 (catches prompt changes without a version bump)
    expected_sha = prompt_template_sha256()
    if "prompt_template_sha256" in audit:
        if audit["prompt_template_sha256"] != expected_sha:
            raise RuntimeError(
                "Stale pretokenized cache: prompt_template_sha256 mismatch. "
                f"Expected {expected_sha}, got {audit['prompt_template_sha256']}. "
                "Re-run pretokenization."
            )
    
    # 3. Model / tokenizer ID
    cached_model_id = audit.get("model_id") or audit.get("tokenizer_id")
    if cached_model_id and cached_model_id != config.model.model_id:
        raise RuntimeError(
            f"Stale cache model mismatch. Expected {config.model.model_id!r}, got {cached_model_id!r}."
        )
    
    # 4. Max tokens (if present in audit and config)
    if "max_tokens" in audit:
        audit_max = int(audit["max_tokens"])
        config_max = int(config.model.max_context_tokens or 2048)
        if audit_max != config_max:
            raise RuntimeError(
                f"Stale cache max_tokens mismatch. Expected {config_max}, got {audit_max}. "
                "Re-run pretokenization."
            )
    
    return audit


def collect_lora_telemetry(model: Any, initial_state: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    import numpy as np
    telemetry: dict[str, Any] = {
        "global_lora_l2": 0.0,
        "global_lora_delta_l2": 0.0,
        "per_layer": {},
        "top_changed_layers": []
    }
    
    layer_deltas = []
    total_l2_sq = 0.0
    total_delta_sq = 0.0
    
    for name, param in model.parameters().items():
        if "lora" not in name.lower():
            continue
            
        # extract layer ID
        layer_id = "unknown"
        import re
        m = re.search(r"layers\.(\d+)", name)
        if m:
            layer_id = m.group(1)
            
        arr = np.array(param)
        l2 = float(np.sqrt((arr * arr).sum()))
        total_l2_sq += l2 * l2
        
        delta_l2 = 0.0
        if initial_state and name in initial_state:
            delta_arr = arr - initial_state[name]
            delta_l2 = float(np.sqrt((delta_arr * delta_arr).sum()))
            total_delta_sq += delta_l2 * delta_l2
            
        if layer_id not in telemetry["per_layer"]:
            telemetry["per_layer"][layer_id] = {"l2": 0.0, "delta_l2": 0.0}
            
        telemetry["per_layer"][layer_id]["l2"] += l2
        telemetry["per_layer"][layer_id]["delta_l2"] += delta_l2
        
    telemetry["global_lora_l2"] = float(np.sqrt(total_l2_sq))
    telemetry["global_lora_delta_l2"] = float(np.sqrt(total_delta_sq))
    
    # Sort layers by delta L2
    layer_list = [(lid, data["delta_l2"]) for lid, data in telemetry["per_layer"].items()]
    layer_list.sort(key=lambda x: x[1], reverse=True)
    telemetry["top_changed_layers"] = [lid for lid, _ in layer_list[:10]]
    
    return telemetry


@dataclass(slots=True)
class TrainingPlan:
    dataset_path: Path
    val_dataset_path: Path | None
    output_path: Path
    dry_run: bool
    args: argparse.Namespace
    warnings: list[str]


@dataclass(slots=True)
class CompletionMaskPreflightResult:
    scanned_rows: int
    marker_hit_rows: int
    marker_hit_rate: float
    mask_zero_fraction_mean: float
    mask_zero_fraction_min: float
    mask_zero_fraction_max: float
    marker_sequences: tuple[tuple[int, ...], ...]


@dataclass(slots=True)
class CoverageState:
    run_id: str
    dataset_fingerprint: dict[str, Any]
    config_fingerprint: str
    total_examples: int
    target_steps: int
    order_mode: str
    order_seed_base: int
    epoch: int
    global_step: int
    batch_cursor_in_epoch: int
    seen_in_epoch_count: int
    next_example_index: int
    epoch_order_checksum: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "total_examples": self.total_examples,
            "target_steps": self.target_steps,
            "order_mode": self.order_mode,
            "order_seed_base": self.order_seed_base,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "batch_cursor_in_epoch": self.batch_cursor_in_epoch,
            "seen_in_epoch_count": self.seen_in_epoch_count,
            "next_example_index": self.next_example_index,
            "epoch_order_checksum": self.epoch_order_checksum,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoverageState":
        return cls(
            run_id=str(data["run_id"]),
            dataset_fingerprint=dict(data["dataset_fingerprint"]),
            config_fingerprint=str(data["config_fingerprint"]),
            total_examples=int(data["total_examples"]),
            target_steps=int(data["target_steps"]),
            order_mode=str(data["order_mode"]),
            order_seed_base=int(data["order_seed_base"]),
            epoch=int(data["epoch"]),
            global_step=int(data["global_step"]),
            batch_cursor_in_epoch=int(data["batch_cursor_in_epoch"]),
            seen_in_epoch_count=int(data["seen_in_epoch_count"]),
            next_example_index=int(data["next_example_index"]),
            epoch_order_checksum=str(data["epoch_order_checksum"]),
            updated_at=str(data["updated_at"]),
        )


class StrictCoverageTracker:
    """Orchestrates strict, order-preserving dataset coverage for SFT training.

    Manages epoch-shuffled example orders, global step cursors, and run locks
    to ensure reproducible and resumable training runs without sample duplication.
    """
    def __init__(
        self,
        *,
        config: PipelineConfig,
        run_id: str,
        run_dir: Path,
        dataset_fingerprint: dict[str, Any],
        config_fingerprint: str,
        total_examples: int,
        target_steps: int,
        resume_requested: bool,
    ) -> None:
        self._config = config
        self._coverage = config.training.coverage
        self._run_id = run_id
        self._run_dir = run_dir
        self._state_path = run_dir / self._coverage.state_file_name
        self._orders_dir = run_dir / self._coverage.epoch_orders_dir_name
        self._dataset_fingerprint = dataset_fingerprint
        self._config_fingerprint = config_fingerprint
        self._total_examples = total_examples
        self._target_steps = target_steps
        self._current_order: list[int] | None = None

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._orders_dir.mkdir(parents=True, exist_ok=True)

        self.state = self._load_or_initialize_state(resume_requested=resume_requested)

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def lock_path(self) -> Path:
        return self._run_dir / self._coverage.lock_file_name

    @property
    def remaining_steps(self) -> int:
        return max(self.state.target_steps - self.state.global_step, 0)

    def peek_next_example_index(self) -> int:
        self._ensure_order_loaded()
        assert self._current_order is not None  # for type-checkers
        if self.state.batch_cursor_in_epoch >= len(self._current_order):
            self._advance_epoch()
        assert self._current_order is not None  # for type-checkers
        return int(self._current_order[self.state.batch_cursor_in_epoch])

    def mark_batch_complete(self, example_index: int) -> None:
        expected = self.peek_next_example_index()
        if example_index != expected:
            raise RuntimeError(
                "Coverage cursor mismatch: "
                f"expected example_index={expected}, got {example_index}."
            )

        self.state.batch_cursor_in_epoch += 1
        self.state.seen_in_epoch_count += 1
        self.state.global_step += 1

        if self.state.batch_cursor_in_epoch >= self.state.total_examples:
            if self.state.seen_in_epoch_count != self.state.total_examples:
                raise RuntimeError(
                    "Epoch coverage invariant violated: "
                    f"seen={self.state.seen_in_epoch_count} expected={self.state.total_examples}."
                )
            self._advance_epoch()
        else:
            self.state.next_example_index = self.peek_next_example_index()

        if self.state.global_step % self._coverage.save_interval_steps == 0:
            self.save(force=True)

    def save(self, *, force: bool = False) -> None:
        if not force and not self._coverage.enabled:
            return
        self.state.updated_at = utc_now_iso8601()
        payload = self.state.to_dict()
        tmp_path = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self._state_path)

    def _advance_epoch(self) -> None:
        self.state.epoch += 1
        self.state.batch_cursor_in_epoch = 0
        self.state.seen_in_epoch_count = 0
        self._load_epoch_order(epoch=self.state.epoch, expected_checksum=None)
        self.state.next_example_index = self.peek_next_example_index()

    def _ensure_order_loaded(self) -> None:
        if self._current_order is None:
            self._load_epoch_order(epoch=self.state.epoch, expected_checksum=self.state.epoch_order_checksum)

    def _load_epoch_order(self, *, epoch: int, expected_checksum: str | None) -> None:
        order_path = self._orders_dir / f"epoch_{epoch:06d}.json"
        if order_path.exists():
            with order_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            order = [int(value) for value in payload.get("order", [])]
        else:
            order = build_epoch_example_order(
                total_examples=self._total_examples,
                order_mode=self._coverage.order_mode,
                seed_base=self._coverage.order_seed_base,
                epoch=epoch,
            )
            with order_path.open("w", encoding="utf-8") as handle:
                json.dump({"epoch": epoch, "order": order}, handle, indent=2, sort_keys=True)

        if len(order) != self._total_examples:
            raise RuntimeError(
                f"Epoch order length mismatch for epoch {epoch}: "
                f"expected {self._total_examples}, got {len(order)}."
            )

        checksum = compute_example_order_checksum(order)
        if expected_checksum and expected_checksum != checksum:
            raise RuntimeError(
                f"Epoch order checksum mismatch for epoch {epoch}: "
                f"state={expected_checksum}, actual={checksum}."
            )

        self._current_order = order
        self.state.epoch_order_checksum = checksum

    def _load_or_initialize_state(self, *, resume_requested: bool) -> CoverageState:
        if self._state_path.exists():
            if not resume_requested and self._coverage.require_state_for_resume:
                raise RuntimeError(
                    "Coverage state exists for this run but no adapter resume path was provided. "
                    "In strict mode, resume must provide both adapter and coverage state."
                )

            with self._state_path.open("r", encoding="utf-8") as handle:
                loaded = CoverageState.from_dict(json.load(handle))

            if loaded.run_id != self._run_id:
                raise RuntimeError(
                    f"Coverage state run_id mismatch: expected {self._run_id}, found {loaded.run_id}."
                )
            if loaded.total_examples != self._total_examples:
                raise RuntimeError(
                    f"Coverage state total_examples mismatch: expected {self._total_examples}, "
                    f"found {loaded.total_examples}."
                )
            if self._coverage.strict_fingerprint:
                if loaded.dataset_fingerprint != self._dataset_fingerprint:
                    raise RuntimeError("Dataset fingerprint mismatch in strict coverage mode.")
                if loaded.config_fingerprint != self._config_fingerprint:
                    raise RuntimeError("Training config fingerprint mismatch in strict coverage mode.")

            loaded.target_steps = self._target_steps
            loaded.order_mode = self._coverage.order_mode
            loaded.order_seed_base = self._coverage.order_seed_base

            if loaded.batch_cursor_in_epoch < 0 or loaded.batch_cursor_in_epoch > self._total_examples:
                raise RuntimeError(
                    "Coverage cursor is out of range: "
                    f"batch_cursor_in_epoch={loaded.batch_cursor_in_epoch}, total_examples={self._total_examples}."
                )

            self.state = loaded
            self._load_epoch_order(epoch=loaded.epoch, expected_checksum=loaded.epoch_order_checksum)
            if loaded.batch_cursor_in_epoch >= self._total_examples:
                self._advance_epoch()
            loaded.next_example_index = self.peek_next_example_index()
            return loaded

        if resume_requested and self._coverage.require_state_for_resume:
            raise RuntimeError(
                "Resume adapter was provided but coverage state is missing in strict mode. "
                f"Expected state file: {self._state_path}"
            )

        self.state = CoverageState(
            run_id=self._run_id,
            dataset_fingerprint=self._dataset_fingerprint,
            config_fingerprint=self._config_fingerprint,
            total_examples=self._total_examples,
            target_steps=self._target_steps,
            order_mode=self._coverage.order_mode,
            order_seed_base=self._coverage.order_seed_base,
            epoch=0,
            global_step=0,
            batch_cursor_in_epoch=0,
            seen_in_epoch_count=0,
            next_example_index=0,
            epoch_order_checksum="",
            updated_at=utc_now_iso8601(),
        )
        self._load_epoch_order(epoch=0, expected_checksum=None)
        self.state.next_example_index = self.peek_next_example_index()
        self.save(force=True)
        return self.state


def _resolve_run_id(output_path: Path, requested_run_id: str | None) -> str:
    if requested_run_id:
        return requested_run_id
    stem = output_path.stem.strip()
    return stem if stem else "stage1"


def _compute_training_config_fingerprint(config: PipelineConfig, plan: TrainingPlan) -> str:
    payload = {
        "model_id": config.model.model_id,
        "max_context_tokens": config.model.max_context_tokens,
        "batch_size": plan.args.batch_size,
        "gradient_accumulation_steps": plan.args.gradient_accumulation_steps,
        "learning_rate": plan.args.learning_rate,
        "epochs": plan.args.epochs,
        "steps_per_save": plan.args.steps_per_save,
        "train_on_completions": plan.args.train_on_completions,
        "lora_rank": plan.args.lora_rank,
        "lora_alpha": plan.args.lora_alpha,
        "lora_dropout": plan.args.lora_dropout,
        "coverage": {
            "order_mode": config.training.coverage.order_mode,
            "order_seed_base": config.training.coverage.order_seed_base,
            "save_interval_steps": config.training.coverage.save_interval_steps,
        },
        "dataset_path": str(plan.dataset_path.resolve()),
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _resolved_training_config_snapshot(config: PipelineConfig, plan: TrainingPlan) -> dict[str, Any]:
    return {
        "model_id": config.model.model_id,
        "max_context_tokens": config.model.max_context_tokens,
        "dataset_path": str(plan.dataset_path.resolve()),
        "learning_rate": plan.args.learning_rate,
        "lora_rank": plan.args.lora_rank,
        "lora_alpha": plan.args.lora_alpha,
        "lora_dropout": plan.args.lora_dropout,
        "lora_num_layers": plan.args.lora_num_layers,
        "epochs": plan.args.epochs,
        "iters": plan.args.iters,
        "train_on_completions": plan.args.train_on_completions,
        "reward_weighted_loss": config.training.reward_weighted_loss,
        "syntax_weighted_loss": config.training.syntax_weighted_loss,
        "pretokenized_packed_cache_path": (
            str(config.training.pretokenized_packed_cache_path)
            if config.training.pretokenized_packed_cache_path
            else None
        ),
        "checkpoint_pin_iterations": list(config.training.checkpoint_pin_iterations),
    }


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _clear_stale_run_lock(lock_path: Path, run_id: str) -> bool:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    lock_run_id = payload.get("run_id")
    if isinstance(lock_run_id, str) and lock_run_id != run_id:
        return False

    lock_host = payload.get("host")
    if not isinstance(lock_host, str) or lock_host != socket.gethostname():
        return False

    lock_pid = payload.get("pid")
    if isinstance(lock_pid, bool):
        return False
    try:
        pid = int(lock_pid)
    except (TypeError, ValueError):
        return False

    if _pid_is_running(pid):
        return False

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _acquire_run_lock(lock_path: Path, run_id: str) -> None:
    """Acquire a file-based lock for the current run_id to prevent parallel training.

    If a stale lock from the same host is detected (PID no longer running), it is cleared.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_payload = {
        "run_id": run_id,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "acquired_at": utc_now_iso8601(),
    }
    fd: int | None = None
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if not _clear_stale_run_lock(lock_path, run_id):
                raise RuntimeError(
                    f"Another trainer appears active for run_id={run_id}. Existing lock: {lock_path}"
                ) from exc

    if fd is None:
        raise RuntimeError(
            f"Unable to acquire run lock for run_id={run_id}. Existing lock: {lock_path}"
        )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(lock_payload, handle, indent=2, sort_keys=True)
    except Exception:
        try:
            lock_path.unlink(missing_ok=True)
        finally:
            raise


def _release_run_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


def _write_run_metadata(
    *,
    tracker: StrictCoverageTracker,
    plan: TrainingPlan,
    config: PipelineConfig,
) -> None:
    frozen_config_path = tracker.run_dir / "frozen_config.yaml"
    shutil.copy2(config.config_path, frozen_config_path)

    metadata = {
        "run_id": tracker.state.run_id,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": utc_now_iso8601(),
        "config_path": str(config.config_path),
        "frozen_config_path": str(frozen_config_path),
        "dataset_path": str(plan.dataset_path.resolve()),
        "val_dataset_path": str(plan.val_dataset_path.resolve()) if plan.val_dataset_path is not None else None,
        "output_path": str(plan.output_path.resolve()),
        "resume_adapter_path": plan.args.adapter_path,
        "dataset_fingerprint": tracker.state.dataset_fingerprint,
        "config_fingerprint": tracker.state.config_fingerprint,
        "named_checkpoint_dir": str((tracker.run_dir / "named_checkpoints").resolve()),
    }
    metadata_path = tracker.run_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _dataset_loader_spec(dataset_path: Path) -> tuple[str, dict[str, str] | None]:
    """Resolve the HF-compatible dataset loader spec (json/parquet/csv)."""
    suffix = dataset_path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return "json", {"train": str(dataset_path)}
    if suffix == ".parquet":
        return "parquet", {"train": str(dataset_path)}
    return str(dataset_path), None


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


def _list_stage1_checkpoints(checkpoint_dir: Path, run_id: str | None = None) -> list[Path]:
    checkpoints: list[Path] = []
    for path in checkpoint_dir.glob("*_adapters.safetensors"):
        if not path.is_file():
            continue
        if run_id is not None and _checkpoint_run_id(path) != run_id:
            continue
        checkpoints.append(path)
    checkpoints.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return checkpoints


def _prune_stage1_checkpoints(
    checkpoint_dir: Path,
    keep_last: int,
    run_id: str | None = None,
    *,
    pin_iterations: frozenset[int] | None = None,
) -> list[Path]:
    if keep_last <= 0:
        return []
    pins = pin_iterations or frozenset()
    checkpoints = _list_stage1_checkpoints(checkpoint_dir, run_id=run_id)
    stale_checkpoints = checkpoints[:-keep_last]
    deleted: list[Path] = []
    for checkpoint in stale_checkpoints:
        iteration = extract_iteration_from_checkpoint_name(checkpoint)
        if iteration is not None and iteration in pins:
            continue
        try:
            checkpoint.unlink()
            checkpoint_metadata_path(checkpoint).unlink(missing_ok=True)
            deleted.append(checkpoint)
        except FileNotFoundError:
            continue
    return deleted


def _prepare_resume_adapter_directory(config: PipelineConfig, resume_adapter_path: Path) -> Path:
    if resume_adapter_path.is_dir():
        return resume_adapter_path
    if resume_adapter_path.suffix.lower() != ".safetensors":
        return resume_adapter_path

    adapter_config_source = config.paths.runs_dir / "adapter_config.json"
    if not adapter_config_source.exists():
        raise RuntimeError(
            "Cannot resume from a safetensors checkpoint because "
            f"`{adapter_config_source}` is missing."
        )

    path_hash = hashlib.md5(str(resume_adapter_path.resolve()).encode()).hexdigest()[:8]
    adapter_dir = config.paths.runs_dir / f"{resume_adapter_path.stem}_{path_hash}_adapter_dir"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    adapter_config_target = adapter_dir / "adapter_config.json"
    adapters_target = adapter_dir / "adapters.safetensors"
    shutil.copy2(adapter_config_source, adapter_config_target)
    if adapters_target.exists() or adapters_target.is_symlink():
        adapters_target.unlink()
    try:
        os.link(resume_adapter_path, adapters_target)
    except OSError:
        shutil.copy2(resume_adapter_path, adapters_target)
    return adapter_dir


def _resolve_resume_adapter_path(
    config: PipelineConfig,
    resume_adapter: Path | None,
    checkpoint_dir: Path,
    dry_run: bool,
    warnings: list[str],
    run_id: str | None = None,
) -> Path | None:
    resolved_resume = resume_adapter
    if resolved_resume is None and not dry_run and config.training.auto_resume_latest_checkpoint:
        checkpoints = _list_stage1_checkpoints(checkpoint_dir, run_id=run_id)
        if checkpoints:
            resolved_resume = checkpoints[-1]
            warnings.append(f"Auto-resuming from latest checkpoint: {resolved_resume}")
    if resolved_resume is None:
        return None
    if not resolved_resume.exists():
        return resolved_resume
    return _prepare_resume_adapter_directory(config, resolved_resume)


def _checkpoint_cleanup_loop(
    checkpoint_dir: Path,
    keep_last: int,
    interval_seconds: int,
    stop_event: threading.Event,
    run_id: str | None,
    pin_iterations: frozenset[int] | None = None,
) -> None:
    while True:
        _prune_stage1_checkpoints(
            checkpoint_dir, keep_last, run_id=run_id, pin_iterations=pin_iterations
        )
        if stop_event.wait(interval_seconds):
            break


def _count_local_dataset_records(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count
    return None


from .harden import (
    _CRITIC_HARD_VIOLATIONS,
    _extract_reference_code_from_record,
    _filter_dataset_by_static_critic,
    _set_completion_in_record,
)
from .harden import _compile_repair_dataset


def _measure_validation_compilation_rate(
    *,
    config: PipelineConfig,
    val_dataset: Any,
    probe_limit: int,
    run_id: str,
) -> tuple[float, int, int]:
    compiler = CompilerService(config.compiler)
    total_available = len(val_dataset)
    probe_count = min(total_available, probe_limit)
    if probe_count <= 0:
        return 0.0, 0, 0

    attempted = 0
    successful = 0
    for row_index in range(probe_count):
        record = val_dataset[row_index]
        code = _extract_reference_code_from_record(record)
        if not code:
            continue
        attempted += 1
        output_dir = config.paths.outputs_dir / "validation_compile_probe" / run_id / f"sample_{row_index:05d}"
        summary = compiler.compile_document(code, output_dir=output_dir, job_name="reference")
        if summary.status == CompileStatus.SUCCESS:
            successful += 1

    if attempted == 0:
        return 0.0, 0, 0
    return successful / attempted, attempted, successful


def build_lora_namespace(
    config: PipelineConfig,
    dataset_path: Path,
    val_dataset_path: Path | None,
    output_path: Path,
    dry_run: bool,
    run_id: str,
    resume_adapter_path: Path | None = None,
    iters: int | None = None,
    save_interval: int = 100,
) -> argparse.Namespace:
    dataset_name, data_files = _dataset_loader_spec(dataset_path)
    if val_dataset_path is not None:
        val_dataset_name, val_data_files = _dataset_loader_spec(val_dataset_path)
        if data_files is None or val_data_files is None:
            raise ValueError(
                "Validation dataset support requires local json/jsonl/parquet files for both train and validation."
            )
        if dataset_name != val_dataset_name:
            raise ValueError("Train and validation datasets must use the same file format.")
        data_files = {**data_files, "validation": str(val_dataset_path)}

    return argparse.Namespace(
        model_path=config.model.model_id,
        dataset=dataset_name if data_files is not None else str(dataset_path),
        data_files=data_files,
        split="train",
        val_split="validation" if val_dataset_path is not None else None,
        val_dataset=str(val_dataset_path) if val_dataset_path is not None else None,
        dataset_config=None,
        batch_size=config.memory.batch_size,
        epochs=None if dry_run else (config.training.epochs if iters is None else None),
        iters=config.training.dry_run_steps if dry_run else (iters if iters is not None else 0),
        learning_rate=config.training.learning_rate,
        steps_per_report=1 if dry_run else 10,
        steps_per_eval=1 if dry_run else config.training.steps_per_eval,
        steps_per_save=1 if dry_run else save_interval,
        val_batches=1 if dry_run else config.training.val_batches,
        max_seq_length=config.model.max_context_tokens,
        lora_rank=config.training.lora_rank,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        lora_num_layers=config.training.lora_num_layers,
        output_path=str(output_path),
        adapter_path=str(resume_adapter_path) if resume_adapter_path is not None else None,
        full_finetune=False,
        train_vision=False,
        grad_checkpoint=config.memory.gradient_checkpointing,
        grad_clip=config.training.max_grad_norm if config.training.max_grad_norm is not None else 1.0,
        train_on_completions=config.training.train_on_completions,
        gradient_accumulation_steps=config.memory.gradient_accumulation_steps,
        assistant_id=config.training.assistant_id if config.training.assistant_id is not None else DEFAULT_ASSISTANT_ID,
        image_resize_shape=list(config.model.image_resize_shape),
        custom_prompt_format=None,
        train_mode="sft",
        run_id=run_id,
    )


def plan_training(
    config: PipelineConfig,
    dataset_path: str | Path | None = None,
    val_dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume_adapter_path: str | Path | None = None,
    run_id: str | None = None,
    dry_run: bool = True,
    require_full_opt_in: bool = True,
    iters: int | None = None,
    save_interval: int = 100,
) -> TrainingPlan:
    ensure_runtime_directories(config)
    if require_full_opt_in:
        require_training_opt_in(config, dry_run=dry_run)

    dataset = Path(dataset_path) if dataset_path else config.training.train_dataset_path
    val_dataset = Path(val_dataset_path) if val_dataset_path else config.training.val_dataset_path
    output = Path(output_path) if output_path else config.paths.runs_dir / "tikz_lora_adapter.safetensors"
    resolved_run_id = _resolve_run_id(output, run_id)
    resume_adapter = (
        Path(resume_adapter_path).expanduser().resolve()
        if resume_adapter_path is not None
        else config.training.resume_adapter_path
    )
    if resume_adapter is not None:
        assert_not_quarantined(resume_adapter)

    warnings: list[str] = []
    if not dataset.exists():
        message = (
            f"Dataset path does not exist: {dataset}. "
            "Training cannot start without a local train dataset."
        )
        if dry_run:
            warnings.append(message)
        else:
            raise RuntimeError(message)
    elif not dry_run:
        train_count = _count_local_dataset_records(dataset)
        if train_count is not None and train_count == 0:
            raise RuntimeError(f"Training dataset is empty: {dataset}")

    if val_dataset is not None and not val_dataset.exists():
        message = (
            f"Validation dataset path does not exist: {val_dataset}. "
            "Validation cannot run without this file."
        )
        if not dry_run and config.training.require_nonempty_validation_dataset:
            raise RuntimeError(message)
        warnings.append(message + " Validation will be disabled until this file exists.")
        val_dataset = None
    elif val_dataset is not None and not dry_run and config.training.require_nonempty_validation_dataset:
        val_count = _count_local_dataset_records(val_dataset)
        if val_count is not None and val_count == 0:
            raise RuntimeError(
                f"Validation dataset is empty: {val_dataset}. "
                "Refuse to start non-dry training with empty validation split."
            )

    if not dry_run and config.training.require_nonempty_gold_eval_dataset:
        gold_eval_path = config.training.gold_eval_dataset_path
        if gold_eval_path is None or not gold_eval_path.exists():
            raise RuntimeError(
                "Gold-eval dataset is required but missing. "
                f"Expected: {gold_eval_path}"
            )
        gold_eval_count = _count_local_dataset_records(gold_eval_path)
        if gold_eval_count is not None and gold_eval_count == 0:
            raise RuntimeError(
                f"Gold-eval dataset is empty: {gold_eval_path}. "
                "Refuse to start non-dry training with empty gold split."
            )

    if config.memory.batch_size != 1:
        warnings.append("Batch size should remain 1 on the 24 GB target machine.")
    if not config.memory.freeze_vision:
        warnings.append("Vision layers should stay frozen for the first local LoRA pass.")
    resume_adapter = _resolve_resume_adapter_path(
        config,
        resume_adapter,
        output.parent,
        dry_run,
        warnings,
        run_id=resolved_run_id,
    )
    if resume_adapter is not None and not resume_adapter.exists():
        warnings.append(
            f"Resume adapter path does not exist yet: {resume_adapter}. "
            "Training cannot resume from a missing adapter checkpoint."
        )

    args = build_lora_namespace(
        config,
        dataset,
        val_dataset,
        output,
        dry_run=dry_run,
        run_id=resolved_run_id,
        resume_adapter_path=resume_adapter,
        iters=iters,
        save_interval=save_interval,
    )
    return TrainingPlan(
        dataset_path=dataset,
        val_dataset_path=val_dataset,
        output_path=output,
        dry_run=dry_run,
        args=args,
        warnings=warnings,
    )


def _import_training_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        import_mlx_core()
        import mlx.optimizers as optim
        from datasets import load_dataset
        from mlx_vlm.lora import TrainingArgs, setup_model_for_training, train, transform_dataset_to_messages
        from mlx_vlm.trainer.datasets import VisionDataset
        from mlx_vlm.utils import load
    except ImportError as exc:
        raise RuntimeError("mlx-vlm training dependencies are required for training.") from exc
    return optim, load_dataset, TrainingArgs, setup_model_for_training, train, transform_dataset_to_messages, VisionDataset, load


def _load_training_dataset(args: argparse.Namespace, load_dataset: Any) -> tuple[Any, Any | None]:
    data_files = getattr(args, "data_files", None)
    if data_files is not None:
        train_file = data_files.get("train") or data_files.get(args.split)
        val_split = getattr(args, "val_split", None)
        val_file = data_files.get("validation") or (data_files.get(val_split) if val_split else None)
        # Load each split independently to avoid PyArrow schema alignment issues
        # when nested fields (e.g. geometry_hints) differ structurally between splits.
        train_dataset = load_dataset(args.dataset, data_files={"train": train_file}, split="train", features=None)
        val_dataset = None
        if val_file:
            val_dataset = load_dataset(args.dataset, data_files={"train": val_file}, split="train", features=None)
        return train_dataset, val_dataset
    if args.dataset_config:
        return load_dataset(args.dataset, args.dataset_config, split=args.split, features=None), None
    return load_dataset(args.dataset, split=args.split, features=None), None



def _resolve_training_iterations(dataset_size: int, args: argparse.Namespace) -> int:
    if getattr(args, "iters", None) is not None and args.iters > 0:
        return args.iters
    if getattr(args, "epochs", None) is not None:
        return math.ceil((dataset_size * args.epochs) / args.batch_size)
    return 0


def _tokenizer_encode_without_special_tokens(tokenizer: Any, text: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return ()

    variants = (
        {"add_special_tokens": False},
        {"add_special_tokens": False, "bos": False, "eos": False},
        {},
    )
    for kwargs in variants:
        try:
            tokens = encode(text, **kwargs)
        except TypeError:
            continue
        if tokens is None:
            return ()
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, (list, tuple)):
            values: list[int] = []
            for token in tokens:
                try:
                    values.append(int(token))
                except (TypeError, ValueError):
                    return ()
            return tuple(values)
    return ()


def _flatten_message_texts(messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        flattened.append({"role": str(message.get("role", "")), "content": str(content)})
    return flattened


def _build_syntax_weight_lookup(
    tokenizer: Any,
    *,
    structural_weight: float,
    command_weight: float,
    coordinate_weight: float,
) -> np.ndarray:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        max_token_id = max((int(token_id) for token_id in vocab.values()), default=0)
    else:
        max_token_id = max(int(getattr(tokenizer, "vocab_size", 0)) - 1, 0)

    lookup = np.ones((max_token_id + 1,), dtype=np.float16)
    for token_id in range(max_token_id + 1):
        try:
            text = tokenizer.decode([token_id]).lstrip("\u2581").strip()
        except Exception:
            continue
        if not text:
            continue
        if STRUCTURAL_TOKEN_PATTERN.match(text):
            lookup[token_id] = np.float16(structural_weight)
        elif COMMAND_TOKEN_PATTERN.match(text):
            lookup[token_id] = np.float16(command_weight)
        elif COORDINATE_TOKEN_PATTERN.match(text):
            lookup[token_id] = np.float16(coordinate_weight)
    return lookup


class EnhancedVisionDataset:
    def __init__(
        self,
        hf_dataset: Any,
        config_dict: dict[str, Any],
        processor: Any,
        *,
        image_resize_shape: list[int] | None,
        reward_weighted_loss: bool,
        reward_weight_field: str,
        reward_weight_floor: float,
        reward_weight_ceil: float,
        syntax_weight_lookup: np.ndarray | None,
    ) -> None:
        from mlx_vlm.trainer.datasets import VisionDataset

        self._dataset = hf_dataset
        self._vision_dataset = VisionDataset(
            hf_dataset,
            config_dict,
            processor,
            image_resize_shape=image_resize_shape,
        )
        self._reward_weighted_loss = reward_weighted_loss
        self._reward_weight_field = reward_weight_field
        self._reward_weight_floor = reward_weight_floor
        self._reward_weight_ceil = reward_weight_ceil
        self._syntax_weight_lookup = syntax_weight_lookup

    def __len__(self) -> int:
        return len(self._vision_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        mx = import_mlx_core()

        raw_item = self._dataset[index]
        item = self._vision_dataset.process(raw_item)
        if self._reward_weighted_loss:
            metadata = dict(raw_item.get("metadata", {}))
            sample_weight = float(metadata.get(self._reward_weight_field, 1.0))
            sample_weight = max(self._reward_weight_floor, min(self._reward_weight_ceil, sample_weight))
            item["reward_weight"] = mx.array(np.array(sample_weight, dtype=np.float32))
        if self._syntax_weight_lookup is not None:
            token_ids = np.array(item["input_ids"]).reshape(-1)
            clipped = np.clip(token_ids, 0, len(self._syntax_weight_lookup) - 1)
            weights = self._syntax_weight_lookup[clipped].astype(np.float32, copy=False)
            item["syntax_weight"] = mx.array(weights)
        return item


def _build_assistant_marker_sequences(tokenizer: Any) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    markers: list[tuple[int, ...]] = []
    for text in ASSISTANT_MARKER_TEXT_CANDIDATES:
        encoded = _tokenizer_encode_without_special_tokens(tokenizer, text)
        if not encoded or encoded in seen:
            continue
        seen.add(encoded)
        markers.append(encoded)
    return tuple(markers)


def _find_first_subsequence(tokens: Sequence[int], marker: Sequence[int]) -> int:
    """Return the index of the first occurrence of `marker` in `tokens`, or -1.

    Uses a sliding-window comparison via NumPy for ~10× speedup on long sequences.
    Falls back to pure Python when inputs are too short for vectorized benefit.
    """
    marker_len = len(marker)
    token_len = len(tokens)
    if marker_len == 0 or marker_len > token_len:
        return -1

    # For very short sequences, pure Python is faster than numpy overhead.
    if token_len < 64:
        limit = token_len - marker_len + 1
        marker_tuple = tuple(marker)
        for start in range(limit):
            if tuple(tokens[start : start + marker_len]) == marker_tuple:
                return start
        return -1

    # Vectorized path: create sliding windows and compare against marker.
    arr = np.asarray(tokens, dtype=np.int32)
    marker_arr = np.asarray(marker, dtype=np.int32)
    windows = np.lib.stride_tricks.sliding_window_view(arr, marker_len)
    matches = np.all(windows == marker_arr, axis=1)
    positions = np.flatnonzero(matches)
    return int(positions[0]) if positions.size > 0 else -1


def _compute_assistant_response_indices(
    input_ids: np.ndarray,
    *,
    assistant_id: int,
    marker_sequences: Sequence[Sequence[int]],
) -> np.ndarray:
    batch_size = input_ids.shape[0]
    assistant_response_index = np.full((batch_size,), -1, dtype=np.int32)
    for row_idx in range(batch_size):
        row = input_ids[row_idx]
        row_tokens = [int(value) for value in row.tolist()]

        boundary = -1
        for marker in marker_sequences:
            start = _find_first_subsequence(row_tokens, marker)
            if start >= 0:
                boundary = start + len(marker) - 1
                break

        if boundary < 0:
            positions = np.where(row == assistant_id)[0]
            if positions.size > 0:
                boundary = int(positions[0])

        assistant_response_index[row_idx] = boundary
    return assistant_response_index


def _compute_all_assistant_boundaries(
    input_ids: np.ndarray,
    *,
    assistant_id: int,
    marker_sequences: Sequence[Sequence[int]],
) -> np.ndarray:
    """
    For each row in input_ids, find ALL positions of every assistant-turn start.

    Returns an int32 array of shape (batch_size, MAX_SLOTS=10) with -1 padding.
    Used for packed-sequence loss masking.
    """
    MAX_SLOTS = 10
    batch_size = input_ids.shape[0]
    result = np.full((batch_size, MAX_SLOTS), -1, dtype=np.int32)
    for row_idx in range(batch_size):
        row = input_ids[row_idx]
        row_tokens = [int(v) for v in row.tolist()]
        found: list[int] = []
        # Search for each marker sequence (non-overlapping, left-to-right)
        search_from = 0
        while search_from < len(row_tokens) and len(found) < MAX_SLOTS:
            best_match = -1
            for marker in marker_sequences:
                if not marker:
                    continue
                m = len(marker)
                for start in range(search_from, len(row_tokens) - m + 1):
                    if tuple(row_tokens[start:start + m]) == tuple(marker):
                        pos = start + m - 1  # index of last marker token
                        if best_match < 0 or pos < best_match:
                            best_match = pos
                        break
            if best_match >= 0:
                found.append(best_match)
                search_from = best_match + 1
            else:
                # Fall back: scan for bare assistant_id tokens
                positions = np.where(row[search_from:] == assistant_id)[0]
                if positions.size > 0:
                    abs_pos = int(positions[0]) + search_from
                    found.append(abs_pos)
                    search_from = abs_pos + 1
                else:
                    break
        for j, b in enumerate(found):
            result[row_idx, j] = b
    return result

def _compute_mask_zero_fraction(attention_mask_row: np.ndarray, assistant_boundary_index: int) -> float:
    if attention_mask_row.size <= 1:
        return 0.0

    weight_mask = np.ones_like(attention_mask_row, dtype=np.int32)
    if assistant_boundary_index >= 0:
        capped = min(assistant_boundary_index, weight_mask.shape[0] - 1)
        if capped >= 0:
            weight_mask[: capped + 1] = 0

    shifted_weight_mask = weight_mask[1:]
    return float(np.mean(shifted_weight_mask == 0))


def _build_training_batch(items: list[dict[str, Any]], max_seq_length: int) -> dict[str, Any]:
    mx = import_mlx_core()

    lengths = [
        min(int(np.array(item["input_ids"]).reshape(-1).shape[0]), max_seq_length)
        for item in items
    ]
    max_len = min(max(lengths), max_seq_length)
    pad_to = 32
    padded_len = 1 + pad_to * ((max_len + pad_to - 1) // pad_to)
    padded_len = min(padded_len, max_seq_length)

    input_ids_batch = np.zeros((len(items), padded_len), dtype=np.int32)
    attention_mask_batch = np.zeros((len(items), padded_len), dtype=np.int32)

    for row, item in enumerate(items):
        arr = np.array(item["input_ids"]).reshape(-1)
        length = min(len(arr), padded_len)
        input_ids_batch[row, :length] = arr[:length]

        if "attention_mask" in item:
            mask = np.array(item["attention_mask"]).reshape(-1)
            attention_mask_batch[row, :length] = mask[:length]
        else:
            attention_mask_batch[row, :length] = 1

    pixel_values_batch = None
    if "pixel_values" in items[0] and items[0]["pixel_values"] is not None:
        values = []
        for item in items:
            value = item["pixel_values"]
            if isinstance(value, mx.array) and value.ndim > 0 and value.shape[0] == 1:
                value = value[0]
            values.append(value)
        pixel_values_batch = mx.stack(values)

    batch: dict[str, Any] = {
        "input_ids": mx.array(input_ids_batch),
        "attention_mask": mx.array(attention_mask_batch),
        "pixel_values": pixel_values_batch,
    }

    extra_keys = [
        key
        for key in items[0]
        if key not in ("input_ids", "attention_mask", "pixel_values")
    ]
    for key in extra_keys:
        values = []
        for i, item in enumerate(items):
            value = item[key]
            # If this is a 1D sequence, align it exactly with input_ids_batch.
            if isinstance(value, mx.array) and value.ndim == 1:
                v_len = value.shape[0]
                if v_len > padded_len:
                    value = value[:padded_len]
                elif v_len < padded_len:
                    pad_width = [(0, padded_len - v_len)]
                    value = mx.pad(value, pad_width, mode="constant", constant_values=1.0)
            
            if isinstance(value, mx.array) and value.ndim > 0 and value.shape[0] == 1:
                value = value[0]
            values.append(value)
            
        if values and isinstance(values[0], mx.array):
            try:
                # All values should now have length padded_len if they were sequences.
                batch[key] = mx.stack(values)
            except Exception:
                # Fallback for non-stackable types
                batch[key] = values[0]
        else:
            batch[key] = values[0]

    return batch


def _vision_language_loss_fn_with_marker_sequences(
    model: Any,
    batch: dict[str, Any],
    train_on_completions: bool = False,
    assistant_id: int = DEFAULT_ASSISTANT_ID,
    assistant_marker_sequences: Sequence[Sequence[int]] = (),
) -> Any:
    mx = import_mlx_core()
    nn = import_mlx_nn()

    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_size, seq_length = input_ids.shape

    if train_on_completions:
        # Use pre-computed weight_mask from packed dataset if available (highest precision).
        if "weight_mask" in batch and batch["weight_mask"] is not None:
            weight_mask = batch["weight_mask"][:, 1:]  # Shift for next-token prediction labels
        else:
            # Standard single-boundary path for non-packed data.
            assistant_response_index = _compute_assistant_response_indices(
                np.array(input_ids),
                assistant_id=assistant_id,
                marker_sequences=assistant_marker_sequences,
            )
            range_matrix = mx.repeat(mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0)
            assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(-1, 1)
            weight_mask = mx.where(assistant_mask, mx.zeros_like(attention_mask), mx.ones_like(attention_mask))[:, 1:]
    else:
        weight_mask = None

    reward_weight = batch.get("reward_weight", batch.get("sample_weight"))
    if reward_weight is not None:
        reward_weight = reward_weight[:, 1:] if getattr(reward_weight, "ndim", 0) > 1 else reward_weight

    syntax_weight = batch.get("syntax_weight")
    if syntax_weight is not None:
        syntax_weight = syntax_weight[:, 1:]

    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]
    lengths = mx.sum(attention_mask, axis=1)
    labels = batch["input_ids"][:, 1:]

    # Filter out training-only keys that the model's forward pass does not accept.
    _training_only_keys = {
        "input_ids",
        "pixel_values",
        "attention_mask",
        "weight_mask",
        "boundary_positions",
        "reward_weight",
        "sample_weight",
        "syntax_weight",
    }
    kwargs = {
        key: value
        for key, value in batch.items()
        if key not in _training_only_keys
    }

    outputs = model(input_ids, pixel_values, attention_mask, **kwargs)
    logits = outputs.logits.astype(mx.float32)

    if logits.shape[1] < labels.shape[1]:
        pad_length = labels.shape[1] - logits.shape[1]
        pad_width = ((0, 0), (0, pad_length), (0, 0))
        logits = mx.pad(logits, pad_width, mode="constant", constant_values=-100)
    elif logits.shape[1] > labels.shape[1]:
        logits = logits[:, -labels.shape[1] :, :]

    seq_len = input_ids.shape[1]
    lengths = mx.minimum(lengths, seq_len)
    length_mask = mx.arange(seq_len)[None, :] < lengths[:, None]

    # Combine completion mask with length mask so the denominator counts only
    # the tokens the model is actually trained on, not all non-pad tokens.
    # This ensures the effective learning rate matches the configured value.
    if weight_mask is not None:
        effective_mask = weight_mask * length_mask
    else:
        effective_mask = length_mask

    ce = nn.losses.cross_entropy(logits, labels)
    if reward_weight is not None:
        if getattr(reward_weight, "ndim", 0) == 1:
            reward_weight = reward_weight.reshape(-1, 1)
        ce = ce * reward_weight
    if syntax_weight is not None:
        if syntax_weight.shape[1] != ce.shape[1]:
            if syntax_weight.shape[1] > ce.shape[1]:
                syntax_weight = syntax_weight[:, : ce.shape[1]]
            else:
                pad_width = [(0, 0), (0, ce.shape[1] - syntax_weight.shape[1])]
                syntax_weight = mx.pad(syntax_weight, pad_width, mode="constant", constant_values=1.0)
        ce = ce * syntax_weight
    ce = ce * effective_mask
    return ce.sum() / mx.maximum(effective_mask.sum(), 1)


def _run_completion_mask_preflight(
    train_dataset: Any,
    *,
    max_seq_length: int,
    assistant_id: int,
    marker_sequences: Sequence[Sequence[int]],
    sample_rows: int,
) -> CompletionMaskPreflightResult:
    if sample_rows <= 0:
        raise RuntimeError("Completion mask preflight requires sample_rows > 0.")

    total_rows = min(len(train_dataset), sample_rows)
    if total_rows <= 0:
        raise RuntimeError("Completion mask preflight requires a non-empty training dataset.")

    marker_hit_rows = 0
    zero_fractions: list[float] = []
    for row_index in range(total_rows):
        item = train_dataset[row_index]
        batch = _build_training_batch([item], max_seq_length=max_seq_length)
        input_ids = np.array(batch["input_ids"])
        attention_mask = np.array(batch["attention_mask"])
        assistant_response_index = _compute_assistant_response_indices(
            input_ids,
            assistant_id=assistant_id,
            marker_sequences=marker_sequences,
        )
        boundary = int(assistant_response_index[0])
        if boundary >= 0:
            marker_hit_rows += 1
        zero_fractions.append(_compute_mask_zero_fraction(attention_mask[0], boundary))

    marker_hit_rate = float(marker_hit_rows) / float(total_rows)
    mask_zero_fraction_mean = float(sum(zero_fractions) / len(zero_fractions))

    return CompletionMaskPreflightResult(
        scanned_rows=total_rows,
        marker_hit_rows=marker_hit_rows,
        marker_hit_rate=marker_hit_rate,
        mask_zero_fraction_mean=mask_zero_fraction_mean,
        mask_zero_fraction_min=min(zero_fractions),
        mask_zero_fraction_max=max(zero_fractions),
        marker_sequences=tuple(tuple(int(token) for token in marker) for marker in marker_sequences),
    )


def _build_strict_iterate_batches(
    *,
    tracker: StrictCoverageTracker,
    original_iterate_batches: Any,
) -> Any:
    def _iterate_batches(dataset: Any, batch_size: int, max_seq_length: int, train: bool = False):
        if not train:
            yield from original_iterate_batches(dataset, batch_size, max_seq_length, train=False)
            return

        if batch_size != 1:
            raise RuntimeError("Strict coverage mode currently requires batch_size=1.")

        pending_example_index: int | None = None
        try:
            while True:
                if pending_example_index is not None:
                    tracker.mark_batch_complete(pending_example_index)
                    pending_example_index = None

                example_index = tracker.peek_next_example_index()
                item = dataset[example_index]
                batch = _build_training_batch([item], max_seq_length=max_seq_length)
                pending_example_index = example_index
                yield batch
        finally:
            if pending_example_index is not None:
                tracker.mark_batch_complete(pending_example_index)
            tracker.save(force=True)

    return _iterate_batches


class PreTokenizedDataset:
    def __init__(
        self,
        tokenized_data: np.ndarray,
        config_dict: dict,
        processor: Any,
        *,
        reward_weights: np.ndarray | None = None,
        syntax_weights: np.ndarray | None = None,
        syntax_weight_lookup: np.ndarray | None = None,
    ):
        self.tokenized_data = tokenized_data
        self.config_dict = config_dict
        self.processor = processor
        self.reward_weights = reward_weights
        self.syntax_weights = syntax_weights
        self.syntax_weight_lookup = syntax_weight_lookup

    def __getitem__(self, index):
        mx = import_mlx_core()
        tokens = self.tokenized_data[index]
        item: dict[str, Any] = {
            "input_ids": mx.array(tokens),
            "pixel_values": None,
        }
        if self.reward_weights is not None:
            # reward_weights for pretokenized cache may be stored as a 1-D array
            # of per-sample scalars or as an object array; handle both.
            rw = self.reward_weights[index]
            item["reward_weight"] = mx.array(np.array(rw, dtype=np.float32))
        if self.syntax_weights is not None:
            sw = self.syntax_weights[index]
            item["syntax_weight"] = mx.array(np.array(sw, dtype=np.float32))
        elif self.syntax_weight_lookup is not None:
            token_ids = np.array(tokens).reshape(-1)
            clipped = np.clip(token_ids, 0, len(self.syntax_weight_lookup) - 1)
            weights = self.syntax_weight_lookup[clipped].astype(np.float32, copy=False)
            item["syntax_weight"] = mx.array(weights)
        return item

    def __len__(self):
        return len(self.tokenized_data)


class PackedPreTokenizedDataset:
    """
    Dataset of pre-packed sequences produced by tools/pack_tokenized_dataset.py.

    Each row is a fixed-length (max_tokens,) array of token IDs.  The matching
    boundaries array records the absolute positions of each assistant-turn start
    within each pack row, padded with -1 for unused slots.

    NOTE: Standard causal attention is used, so completion tokens can attend to
    prior examples in the same pack.  This is accepted technical debt for a 1-epoch
    run; the correct fix is block-diagonal attention masking.
    """

    def __init__(
        self,
        packed_ids: np.ndarray,
        boundaries: np.ndarray,
        packed_masks: np.ndarray,
        reward_weights: np.ndarray | None = None,
        syntax_weights: np.ndarray | None = None,
        syntax_weight_lookup: np.ndarray | None = None,
    ):
        assert packed_ids.shape[0] == boundaries.shape[0] == packed_masks.shape[0], (
            "packed_ids, boundaries, and packed_masks must have the same number of rows"
        )
        if reward_weights is not None:
            assert reward_weights.shape == packed_ids.shape, (
                "reward_weights must match packed_ids shape"
            )
        if syntax_weights is not None:
            assert syntax_weights.shape == packed_ids.shape, (
                "syntax_weights must match packed_ids shape"
            )
        self.packed_ids = packed_ids      # (N, max_tokens)  int32
        self.boundaries = boundaries      # (N, MAX_SLOTS)   int32, padded with -1
        self.packed_masks = packed_masks  # (N, max_tokens)  uint8
        self.reward_weights = reward_weights
        self.syntax_weights = syntax_weights
        self.syntax_weight_lookup = syntax_weight_lookup

    def __getitem__(self, index):
        mx = import_mlx_core()
        tokens = self.packed_ids[index]   # 1-D int32 array
        bounds = self.boundaries[index]   # 1-D int32 array (padded)
        mask = self.packed_masks[index]   # 1-D uint8 array
        # boundary_positions travels through _build_training_batch as an extra key
        batch = {
            "input_ids": mx.array(tokens),
            "pixel_values": None,
            "boundary_positions": mx.array(bounds),
            "weight_mask": mx.array(mask).astype(mx.float32),
        }
        if self.reward_weights is not None:
            batch["reward_weight"] = mx.array(self.reward_weights[index]).astype(mx.float32)
        if self.syntax_weights is not None:
            batch["syntax_weight"] = mx.array(self.syntax_weights[index]).astype(mx.float32)
        elif self.syntax_weight_lookup is not None:
            syntax_weights = self.syntax_weight_lookup[np.clip(tokens, 0, len(self.syntax_weight_lookup) - 1)]
            batch["syntax_weight"] = mx.array(syntax_weights).astype(mx.float32)
        return batch

    def __len__(self):
        return len(self.packed_ids)


def _validate_lora_unwrap_and_optimizer_state(
    *,
    model: Any,
    optimizer: Any,
    cutoff: int,
) -> tuple[int, int]:
    from mlx.utils import tree_flatten
    from mlx_vlm.trainer.lora import LoRaLayer

    for name, module in model.language_model.named_modules():
        layer_idx = extract_layer_index(name)
        if layer_idx is not None and layer_idx < cutoff and isinstance(module, LoRaLayer):
            raise RuntimeError(f"Adapter found in frozen layer {layer_idx}: {name}")

    trainable_tree = model.trainable_parameters()
    trainable_paths = {
        path for path, value in tree_flatten(trainable_tree)
        if hasattr(value, "shape")
    }
    optimizer.init(trainable_tree)
    optimizer_state_paths = {
        path for path, _ in tree_flatten(optimizer.state)
        if path not in {"step", "learning_rate"}
    }
    expected_optimizer_paths = {
        suffix
        for path in trainable_paths
        for suffix in (f"{path}.m", f"{path}.v")
    }
    if optimizer_state_paths != expected_optimizer_paths:
        unexpected = sorted(optimizer_state_paths - expected_optimizer_paths)[:5]
        missing = sorted(expected_optimizer_paths - optimizer_state_paths)[:5]
        raise RuntimeError(
            "Optimizer state mismatch after LoRA layer limiting. "
            f"unexpected={unexpected}, missing={missing}."
        )

    return len(trainable_paths), len(optimizer_state_paths)


def _execute_training(
    config: PipelineConfig,
    plan: TrainingPlan,
    *,
    limit_examples: int | None = None,
    forced_iters: int | None = None,
) -> TrainingPlan:
    optim, load_dataset, TrainingArgs, setup_model_for_training, _, transform_dataset_to_messages, VisionDataset, load = (
        _import_training_runtime()
    )

    coverage_tracker: StrictCoverageTracker | None = None
    lock_acquired = False
    checkpoint_cleanup_stop: threading.Event | None = None
    checkpoint_cleanup_thread: threading.Thread | None = None
    sft_trainer_module: Any | None = None
    original_iterate_batches: Any | None = None
    original_save_adapter: Any | None = None
    original_evaluate: Any | None = None
    named_checkpoint_policy: NamedCheckpointPolicyManager | None = None
    latest_val_loss: float | None = None
    dataset_snapshot_id_for_checkpoints: str | None = None
    config_fingerprint_for_checkpoints: str | None = None
    steps_per_epoch: int | None = None
    resume_weights_path: Path | None = None
    packed_audit_payload: dict[str, Any] | None = None

    try:
        if config.training.pretokenized_packed_cache_path:
            packed_audit_payload = _load_and_validate_pack_audit(
                packed_path=config.training.pretokenized_packed_cache_path,
                assistant_id=plan.args.assistant_id,
                min_marker_hit_rate=config.training.completion_mask_preflight_min_marker_hit_rate,
                min_mask_zero_fraction=config.training.completion_mask_preflight_min_mask_zero_fraction,
                reward_weight_path=config.training.reward_weight_path,
                syntax_weight_path=config.training.syntax_weight_path,
            )

        model, processor = load(
            plan.args.model_path,
            processor_config={"trust_remote_code": True},
        )

        dataset, val_dataset = _load_training_dataset(plan.args, load_dataset)
        if len(dataset) < config.memory.batch_size:
            raise RuntimeError(
                f"Training dataset must contain at least {config.memory.batch_size} example(s); found {len(dataset)}."
            )
        if limit_examples is not None:
            capped_examples = min(len(dataset), max(limit_examples, config.memory.batch_size))
            dataset = dataset.select(range(capped_examples))

        steps_per_epoch = max(1, math.ceil(len(dataset) / plan.args.batch_size))
        if plan.dataset_path.exists():
            dataset_snapshot_id_for_checkpoints = compute_dataset_fingerprint(plan.dataset_path).sha256
        config_fingerprint_for_checkpoints = _compute_training_config_fingerprint(config, plan)

        if val_dataset is not None and plan.val_dataset_path is not None:
            val_path = plan.val_dataset_path
            if val_path.suffix.lower() in {".jsonl", ".json"}:
                vlengths = iter_jsonl_metadata_token_lengths(val_path)
                profile = summarize_token_lengths(vlengths)
                plan.warnings.append(
                    "Validation metadata token_length profile: " + json.dumps(profile, sort_keys=True)
                )
                ctx = float(config.model.max_context_tokens)
                frac = config.training.val_metadata_length_warn_fraction_of_context
                if profile["count"] > 0 and float(profile["p95"]) < frac * ctx:
                    plan.warnings.append(
                        "Validation p95 metadata token_length "
                        f"({profile['p95']:.0f}) is below {frac:.2f} * max_context_tokens "
                        f"({ctx:.0f}); validation loss may not stress long-context behaviour."
                    )

        if val_dataset is not None and config.training.min_validation_compilation_rate > 0.0:
            run_id = str(getattr(plan.args, "run_id", "stage1"))
            observed_rate, attempted, successful = _measure_validation_compilation_rate(
                config=config,
                val_dataset=val_dataset,
                probe_limit=config.training.validation_compile_probe_limit,
                run_id=run_id,
            )
            if attempted <= 0:
                raise RuntimeError(
                    "Validation compilation probe could not extract any reference code from validation records."
                )
            plan.warnings.append(
                "Validation compile probe: "
                f"rate={observed_rate:.3f} ({successful}/{attempted})"
            )
            if observed_rate < config.training.min_validation_compilation_rate:
                raise RuntimeError(
                    "Validation compilation rate below configured floor: "
                    f"observed={observed_rate:.3f}, "
                    f"required={config.training.min_validation_compilation_rate:.3f}."
                )

        configured_iters = (
            forced_iters if forced_iters is not None else _resolve_training_iterations(len(dataset), plan.args)
        )

        if config.training.pretokenized_packed_cache_path and config.training.coverage.enabled:
            raise RuntimeError(
                "Packed pre-tokenized cache and strict coverage tracking are mutually exclusive. "
                "Disable coverage tracking or use unpacked data for strict coverage mode."
            )

        if config.training.coverage.enabled:
            if plan.args.batch_size != 1:
                raise RuntimeError("Strict coverage mode currently requires batch_size=1.")
            if "example_index" not in dataset.column_names:
                raise RuntimeError(
                    "Strict coverage mode requires `example_index` in the training dataset. "
                    "Re-run dataset splitting before training."
                )

            example_indices = [int(value) for value in dataset["example_index"]]
            validate_row_aligned_example_indices(example_indices)

            if not plan.dataset_path.exists():
                raise RuntimeError(
                    "Strict coverage mode requires a local dataset path for fingerprinting. "
                    f"Missing dataset file: {plan.dataset_path}"
                )

            run_id = str(plan.args.run_id)
            dataset_fingerprint_obj = compute_dataset_fingerprint(plan.dataset_path)
            dataset_fingerprint = dataset_fingerprint_obj.to_dict()
            dataset_snapshot_id_for_checkpoints = dataset_fingerprint_obj.sha256
            config_fingerprint = config_fingerprint_for_checkpoints
            assert config_fingerprint is not None
            run_dir = config.paths.runs_dir / run_id
            coverage_tracker = StrictCoverageTracker(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                dataset_fingerprint=dataset_fingerprint,
                config_fingerprint=config_fingerprint,
                total_examples=len(dataset),
                target_steps=configured_iters,
                resume_requested=plan.args.adapter_path is not None,
            )
            _acquire_run_lock(coverage_tracker.lock_path, run_id)
            lock_acquired = True
            _write_run_metadata(tracker=coverage_tracker, plan=plan, config=config)

            remaining_iters = coverage_tracker.remaining_steps
            if remaining_iters <= 0:
                plan.warnings.append(
                    "Coverage target already reached; no additional training iterations were scheduled."
                )
                return plan
            configured_iters = remaining_iters
            plan.warnings.append(
                f"Strict coverage enabled: epoch={coverage_tracker.state.epoch} "
                f"global_step={coverage_tracker.state.global_step} "
                f"next_example_index={coverage_tracker.state.next_example_index}"
            )

        plan.args.iters = configured_iters

        run_id = str(plan.args.run_id)
        run_dir_for_named = coverage_tracker.run_dir if coverage_tracker is not None else (config.paths.runs_dir / run_id)
        run_dir_for_named.mkdir(parents=True, exist_ok=True)
        named_checkpoint_policy = NamedCheckpointPolicyManager(
            named_dir=run_dir_for_named / "named_checkpoints",
            stage="stage1",
            run_id=run_id,
        )

        def _checkpoint_context(iteration_hint: int | None = None) -> CheckpointContext:
            epoch: int | None = None
            global_step: int | None = None
            batch_in_epoch: int | None = None
            sample_cursor_in_epoch: int | None = None
            epoch_order_checksum: str | None = None
            dataset_snapshot_id = dataset_snapshot_id_for_checkpoints
            config_fingerprint = config_fingerprint_for_checkpoints

            if coverage_tracker is not None:
                total_examples = max(coverage_tracker.state.total_examples, 1)
                global_step = coverage_tracker.state.global_step
                if iteration_hint is not None and global_step < iteration_hint:
                    global_step = iteration_hint
                epoch = global_step // total_examples
                batch_in_epoch = global_step % total_examples
                sample_cursor_in_epoch = batch_in_epoch
                epoch_order_checksum = coverage_tracker.state.epoch_order_checksum
                dataset_snapshot_id = str(coverage_tracker.state.dataset_fingerprint.get("sha256"))
                config_fingerprint = coverage_tracker.state.config_fingerprint
            elif iteration_hint is not None and steps_per_epoch is not None:
                global_step = iteration_hint
                epoch = global_step // steps_per_epoch
                batch_in_epoch = global_step % steps_per_epoch
                sample_cursor_in_epoch = batch_in_epoch

            return CheckpointContext(
                epoch=epoch,
                global_step=global_step,
                batch_in_epoch=batch_in_epoch,
                sample_cursor_in_epoch=sample_cursor_in_epoch,
                dataset_snapshot_id=dataset_snapshot_id,
                epoch_order_checksum=epoch_order_checksum,
                training_config_fingerprint=config_fingerprint,
            )

        def _is_epoch_boundary(context: CheckpointContext) -> bool:
            if context.global_step is None or context.global_step <= 0:
                return False
            if coverage_tracker is not None:
                return context.global_step % max(coverage_tracker.state.total_examples, 1) == 0
            if steps_per_epoch is not None and steps_per_epoch > 0:
                return context.global_step % steps_per_epoch == 0
            return False

        resume_weights_path = resolve_adapter_weights_path(plan.args.adapter_path)
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

        model_type = getattr(getattr(model, "config", None), "model_type", None)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        syntax_weight_lookup = None
        if config.training.syntax_weighted_loss and config.training.syntax_weight_path is None:
            syntax_weight_lookup = _build_syntax_weight_lookup(
                tokenizer,
                structural_weight=config.training.syntax_structural_weight,
                command_weight=config.training.syntax_command_weight,
                coordinate_weight=config.training.syntax_coordinate_weight,
            )
        if config.training.pretokenized_packed_cache_path:
            packed_path = config.training.pretokenized_packed_cache_path
            boundaries_path = packed_path.with_name(packed_path.stem + "_boundaries.npy")
            masks_path = packed_path.with_name(packed_path.stem + "_masks.npy")
            if not boundaries_path.exists() or not masks_path.exists():
                raise RuntimeError(
                    f"Packed boundaries/masks file not found in {packed_path.parent}. "
                    "Run tools/pack_tokenized_dataset.py to generate them."
                )
                
            _verify_cache_audit(packed_path, config, is_packed=True)
            
            print(f"Loading packed training cache from {packed_path}...")
            packed_ids = np.load(packed_path, mmap_mode="r")
            boundaries = np.load(boundaries_path, mmap_mode="r")
            packed_masks = np.load(masks_path, mmap_mode="r")
            reward_weights = None
            syntax_weights = None
            if config.training.reward_weighted_loss:
                if config.training.reward_weight_path is None:
                    raise RuntimeError(
                        "reward_weighted_loss is enabled with a packed cache, but no reward_weight_path was configured. "
                        "Re-pack the dataset with reward weights or disable packed training for weighted loss."
                    )
                reward_weights = np.load(config.training.reward_weight_path, mmap_mode="r")
            if config.training.syntax_weighted_loss and config.training.syntax_weight_path is not None:
                syntax_weights = np.load(config.training.syntax_weight_path, mmap_mode="r")
            train_dataset = PackedPreTokenizedDataset(
                packed_ids,
                boundaries,
                packed_masks,
                reward_weights=reward_weights,
                syntax_weights=syntax_weights,
                syntax_weight_lookup=syntax_weight_lookup if config.training.syntax_weighted_loss else None,
            )
            if forced_iters is None and plan.args.epochs is not None:
                configured_iters = _resolve_training_iterations(len(train_dataset), plan.args)
                plan.args.iters = configured_iters
                steps_per_epoch = max(1, math.ceil(len(train_dataset) / plan.args.batch_size))
            plan.warnings.append(
                f"Packed dataset: {len(train_dataset)} packs "
                f"(~{len(packed_ids) * packed_ids.shape[1] // 1000}k tokens, "
                f"avg {packed_ids.shape[1]} tokens/pack)"
            )
            if packed_audit_payload is not None:
                plan.warnings.append(
                    "Packed dataset audit: "
                    f"marker_hit_rate={float(packed_audit_payload['marker_hit_rate']):.3f}, "
                    f"mask_zero_fraction={float(packed_audit_payload['mask_zero_fraction']):.3f}, "
                    f"truncated_sequences={int(packed_audit_payload.get('truncated_sequences', 0))}"
                )
        elif config.training.pretokenized_cache_path:
            # Support syntax-weighted loss with a pretokenized cache by loading
            # an optional `syntax_weight_path` (object-array) or falling back to
            # a token-level `syntax_weight_lookup` computed from the tokenizer.
            reward_weights = None
            syntax_weights = None
            if config.training.reward_weighted_loss:
                if config.training.reward_weight_path is None:
                    raise RuntimeError(
                        "reward_weighted_loss is enabled with a pretokenized cache, but no reward_weight_path was configured. "
                        "Provide a reward_weight_path or disable reward_weighted_loss."
                    )
                reward_weights = np.load(config.training.reward_weight_path, allow_pickle=True)
            if config.training.syntax_weighted_loss:
                if config.training.syntax_weight_path is not None:
                    syntax_weights = np.load(config.training.syntax_weight_path, allow_pickle=True)
                else:
                    # syntax_weight_lookup may have been built earlier
                    syntax_weights = None
                    
            _verify_cache_audit(config.training.pretokenized_cache_path, config, is_packed=False)
            
            print(f"Loading pre-tokenized training cache from {config.training.pretokenized_cache_path}...")
            tokenized_data = np.load(config.training.pretokenized_cache_path, allow_pickle=True)
            train_dataset = PreTokenizedDataset(
                tokenized_data,
                model.config.__dict__,
                processor,
                reward_weights=reward_weights,
                syntax_weights=syntax_weights,
                syntax_weight_lookup=syntax_weight_lookup if config.training.syntax_weighted_loss else None,
            )
            if forced_iters is None and plan.args.epochs is not None:
                configured_iters = _resolve_training_iterations(len(train_dataset), plan.args)
                plan.args.iters = configured_iters
                steps_per_epoch = max(1, math.ceil(len(train_dataset) / plan.args.batch_size))
        else:
            # Static critic training gate (plan §2.5): drop records with critical
            # static violations before training. Only applies to the JSONL path;
            # packed/pretokenized paths operate on token IDs where text recovery
            # is not feasible without a full tokenizer round-trip.
            is_raw_list_or_hf = isinstance(dataset, list) or (dataset is not None and "Dataset" in type(dataset).__name__)
            if config.training.static_critic_training_gate and is_raw_list_or_hf:
                if not isinstance(dataset, list):
                    dataset = list(dataset)
                dataset, critic_dropped = _filter_dataset_by_static_critic(
                    dataset,
                    max_violations=config.training.static_critic_max_violations,
                )
                plan.warnings.append(
                    f"Static critic gate: dropped {critic_dropped} records with critical violations "
                    f"(max_violations={config.training.static_critic_max_violations}), "
                    f"{len(dataset)} records remaining."
                )
                if not dataset:
                    raise RuntimeError(
                        "Static critic gate removed all training records. "
                        "Loosen static_critic_max_violations or disable static_critic_training_gate."
                    )
            # Compile-and-repair pre-flight (plan §2.3): normalize → compile →
            # repair each JSONL completion using error line hints before training.
            # Model-free: uses only normalize_tikz() + Tectonic. Runs after the
            # static critic gate so we only attempt repair on passable records.
            if config.training.repair_before_training and is_raw_list_or_hf:
                if not isinstance(dataset, list):
                    dataset = list(dataset)
                print(
                    f"[repair_before_training] Running compile-and-repair pre-flight on "
                    f"{len(dataset)} records (timeout={config.training.repair_before_training_timeout}s)…"
                )
                dataset, repaired_count, kept_original = _compile_repair_dataset(
                    dataset,
                    config=config,
                    timeout_seconds=config.training.repair_before_training_timeout,
                )
                plan.warnings.append(
                    f"Compile-repair pre-flight: repaired={repaired_count}, "
                    f"kept_original={kept_original} (repair failed), "
                    f"total={len(dataset)} records."
                )
            from datasets import Dataset as HFDataset
            if isinstance(dataset, list):
                dataset = HFDataset.from_list(dataset)
            dataset_messages = transform_dataset_to_messages(dataset, model_type, plan.args.custom_prompt_format)


            train_dataset = EnhancedVisionDataset(
                dataset_messages,
                model.config.__dict__,
                processor,
                image_resize_shape=list(config.model.image_resize_shape),
                reward_weighted_loss=config.training.reward_weighted_loss,
                reward_weight_field=config.training.reward_weight_field,
                reward_weight_floor=config.training.reward_weight_floor,
                reward_weight_ceil=config.training.reward_weight_ceil,
                syntax_weight_lookup=syntax_weight_lookup if config.training.syntax_weighted_loss else None,
            )
        assistant_marker_sequences = _build_assistant_marker_sequences(tokenizer)
        val_dataset_messages = None
        if val_dataset is not None:
            if len(val_dataset) < config.memory.batch_size:
                raise RuntimeError(
                    f"Validation dataset must contain at least {config.memory.batch_size} example(s); found {len(val_dataset)}."
                )
            if isinstance(val_dataset, list):
                from datasets import Dataset as HFDataset
                val_dataset = HFDataset.from_list(val_dataset)
            val_dataset_messages = transform_dataset_to_messages(
                val_dataset,
                model_type,
                plan.args.custom_prompt_format,
            )
            val_dataset_messages = VisionDataset(
                val_dataset_messages,
                model.config.__dict__,
                processor,
                image_resize_shape=list(config.model.image_resize_shape),
            )

        if (
            plan.args.train_on_completions
            and config.training.completion_mask_preflight_enabled
            and not config.training.pretokenized_packed_cache_path
        ):
            preflight_result = _run_completion_mask_preflight(
                train_dataset,
                max_seq_length=plan.args.max_seq_length,
                assistant_id=plan.args.assistant_id,
                marker_sequences=assistant_marker_sequences,
                sample_rows=config.training.completion_mask_preflight_rows,
            )
            plan.warnings.append(
                "Completion mask preflight: "
                f"marker_hit_rate={preflight_result.marker_hit_rate:.3f} "
                f"({preflight_result.marker_hit_rows}/{preflight_result.scanned_rows}), "
                f"mask_zero_fraction_mean={preflight_result.mask_zero_fraction_mean:.3f}"
            )
            if (
                preflight_result.marker_hit_rate
                < config.training.completion_mask_preflight_min_marker_hit_rate
            ):
                raise RuntimeError(
                    "Completion mask preflight failed: assistant marker hit-rate below configured floor. "
                    f"observed={preflight_result.marker_hit_rate:.3f}, "
                    f"required={config.training.completion_mask_preflight_min_marker_hit_rate:.3f}."
                )
            if (
                preflight_result.mask_zero_fraction_mean
                < config.training.completion_mask_preflight_min_mask_zero_fraction
            ):
                raise RuntimeError(
                    "Completion mask preflight failed: masked-token fraction below configured floor. "
                    f"observed={preflight_result.mask_zero_fraction_mean:.3f}, "
                    f"required={config.training.completion_mask_preflight_min_mask_zero_fraction:.3f}."
                )

        validate_resumed_adapter_lora_hyperparams(
            adapter_path=plan.args.adapter_path,
            lora_rank=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
        )

        prepared_adapter_path = prepare_adapter_for_mlx_vlm(plan.args.adapter_path)
        model = setup_model_for_training(model, plan.args, prepared_adapter_path)

        # --- LoRA layer limiting ---
        # If lora_num_layers is set, remove LoRA wrappers from early layers so
        # gradients only flow through the last N transformer layers. The forward
        # pass still runs the frozen base weights for all layers.
        lora_num_layers = getattr(plan.args, "lora_num_layers", None)
        total_layers = 0
        cutoff = 0
        if lora_num_layers is not None:
            layer_nums: set[int] = set()
            for name, _ in model.language_model.named_modules():
                layer_idx = extract_layer_index(name)
                if layer_idx is not None:
                    layer_nums.add(layer_idx)
            total_layers = max(layer_nums) + 1 if layer_nums else 0
            cutoff = max(0, total_layers - lora_num_layers)

        trainable_params: int | None = None
        optimizer_state_entries: int | None = None
        if lora_num_layers is not None:
            from mlx_vlm.trainer.lora import LoRaLayer

            unwrapped_count = 0
            for name, module in list(model.language_model.named_modules()):
                if not isinstance(module, LoRaLayer):
                    continue
                layer_num = extract_layer_index(name)
                if layer_num is not None and layer_num < cutoff:
                    # Unwrap: replace the LoRaLayer with its inner base linear
                    unwrap_lora_layer(model.language_model, name, module.original_layer)
                    unwrapped_count += 1

            plan.warnings.append(
                f"LoRA layer limiting: kept last {lora_num_layers}/{total_layers} layers, "
                f"unwrapped {unwrapped_count} LoRA modules from early layers"
            )
        # Use the definitively resolved iters for the scheduler to ensure proper warmup/decay
        total_steps = configured_iters
        warmup_steps = int(0.10 * total_steps)  # 10% warmup; stabilises early-step gradient noise
        # Minimum end LR = 1% of peak to avoid wasting the final ~10% of training steps.
        cosine_end_lr = plan.args.learning_rate * 0.01
        weight_decay = config.training.weight_decay  # default 0.01 per plan recommendation
        if warmup_steps > 0:
            linear = optim.linear_schedule(0.0, plan.args.learning_rate, steps=warmup_steps)
            cosine = optim.cosine_decay(plan.args.learning_rate, total_steps - warmup_steps, end=cosine_end_lr)
            lr_schedule = optim.join_schedules([linear, cosine], [warmup_steps])
            optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
        else:
            cosine = optim.cosine_decay(plan.args.learning_rate, total_steps, end=cosine_end_lr)
            optimizer = optim.AdamW(learning_rate=cosine, weight_decay=weight_decay)
        if lora_num_layers is not None:
            trainable_params, optimizer_state_entries = _validate_lora_unwrap_and_optimizer_state(
                model=model,
                optimizer=optimizer,
                cutoff=cutoff,
            )
            plan.warnings.append(
                "LoRA optimizer audit: "
                f"trainable_parameters={trainable_params}, "
                f"optimizer_state_entries={optimizer_state_entries}"
            )


        telemetry_path = Path(plan.args.output_path).expanduser().resolve().parent / "phase_boundary_telemetry.json"
        write_phase_boundary_telemetry(
            telemetry_path,
            {
                "run_id": str(plan.args.run_id),
                "lora_num_layers": lora_num_layers,
                "total_transformer_layers": total_layers,
                "lora_cutoff_layer_index_exclusive": cutoff,
                "trainable_parameter_leaf_count": trainable_params,
                "optimizer_state_leaf_count": optimizer_state_entries,
                "configured_iters": configured_iters,
                "warmup_steps": warmup_steps,
                "warmup_fraction": 0.1,
                "peak_learning_rate": plan.args.learning_rate,
                "weight_decay": weight_decay,
                "cosine_end_learning_rate": cosine_end_lr,
                "max_seq_length": plan.args.max_seq_length,
                "val_batches": plan.args.val_batches,
                "resume_adapter_path": plan.args.adapter_path,
            },
        )
        plan.warnings.append(f"Phase boundary telemetry written to {telemetry_path}")
        training_args = TrainingArgs(
            batch_size=plan.args.batch_size,
            iters=configured_iters,
            steps_per_report=plan.args.steps_per_report,
            steps_per_eval=plan.args.steps_per_eval,
            steps_per_save=plan.args.steps_per_save,
            val_batches=plan.args.val_batches,
            max_seq_length=plan.args.max_seq_length,
            adapter_file=plan.args.output_path,
            grad_checkpoint=plan.args.grad_checkpoint,
            learning_rate=plan.args.learning_rate,
            grad_clip=plan.args.grad_clip,
            gradient_accumulation_steps=plan.args.gradient_accumulation_steps,
            full_finetune=plan.args.full_finetune,
        )

        checkpoint_dir = Path(plan.args.output_path).expanduser().resolve().parent
        checkpoint_pins = frozenset(config.training.checkpoint_pin_iterations)
        if config.training.checkpoint_keep_last > 0:
            _prune_stage1_checkpoints(
                checkpoint_dir,
                config.training.checkpoint_keep_last,
                run_id=run_id,
                pin_iterations=checkpoint_pins,
            )
            checkpoint_cleanup_stop = threading.Event()
            checkpoint_cleanup_thread = threading.Thread(
                target=_checkpoint_cleanup_loop,
                args=(
                    checkpoint_dir,
                    config.training.checkpoint_keep_last,
                    config.training.checkpoint_cleanup_interval_seconds,
                    checkpoint_cleanup_stop,
                    run_id,
                    checkpoint_pins,
                ),
                daemon=True,
            )
            checkpoint_cleanup_thread.start()

        import mlx_vlm.trainer.sft_trainer as sft_trainer

        # Evaluation milestones: 50% and 100% (final) only.
        total_milestone_iters = (
            coverage_tracker.state.target_steps
            if coverage_tracker is not None
            else configured_iters
        )
        milestones_abs = {max(1, total_milestone_iters // 2), total_milestone_iters}
        if coverage_tracker is not None:
            g0 = coverage_tracker.state.global_step
            tikz_eval_at = frozenset(
                m - g0 for m in milestones_abs if 1 <= (m - g0) <= configured_iters
            )
        else:
            tikz_eval_at = frozenset(m for m in milestones_abs if 1 <= m <= configured_iters)
        if val_dataset_messages is not None:
            if not tikz_eval_at:
                tikz_eval_at = frozenset({configured_iters})
            training_args._tikz_eval_at = tikz_eval_at

        sft_trainer_module = sft_trainer
        if coverage_tracker is not None:
            original_iterate_batches = sft_trainer.iterate_batches
            sft_trainer.iterate_batches = _build_strict_iterate_batches(
                tracker=coverage_tracker,
                original_iterate_batches=original_iterate_batches,
            )

        original_save_adapter = sft_trainer.save_adapter
        original_evaluate = sft_trainer.evaluate

        def _evaluate_wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal latest_val_loss
            result = original_evaluate(*args, **kwargs)
            try:
                latest_val_loss = float(result)
            except (TypeError, ValueError):
                latest_val_loss = None
            return result

        def _save_adapter_wrapper(model_obj: Any, adapter_file: str | Path) -> None:
            assert original_save_adapter is not None
            original_save_adapter(model_obj, adapter_file)

            if named_checkpoint_policy is None:
                return

            checkpoint_path = Path(adapter_file).expanduser().resolve()
            
            # Ensure adapter_config.json exists next to the checkpoint for MLX-VLM compatibility
            adapter_config_path = checkpoint_path.parent / "adapter_config.json"
            adapter_config = {
                "rank": config.training.lora_rank,
                "alpha": config.training.lora_alpha,
                "dropout": config.training.lora_dropout,
                "lora_layers": config.training.lora_num_layers,
                "target_modules": ["q_proj", "v_proj"],
            }
            adapter_config_path.write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")

            if not checkpoint_path.exists():
                return

            iteration = extract_iteration_from_checkpoint_name(checkpoint_path)
            context = _checkpoint_context(iteration_hint=iteration)
            metrics = {"validation_loss": latest_val_loss} if latest_val_loss is not None else None
            checkpoint_role = "periodic_checkpoint" if iteration is not None else "adapter_snapshot"
            extra_metadata = {
                "resolved_training_config": _resolved_training_config_snapshot(config, plan),
                "pack_audit_sha256": (
                    _file_sha256(config.training.pretokenized_packed_cache_path.with_name(
                        config.training.pretokenized_packed_cache_path.stem + "_audit.json"
                    ))
                    if config.training.pretokenized_packed_cache_path
                    and config.training.pretokenized_packed_cache_path.with_name(
                        config.training.pretokenized_packed_cache_path.stem + "_audit.json"
                    ).exists()
                    else None
                ),
                "adapter_sha256": _file_sha256(checkpoint_path),
                "loss_normalization_version": "completion_effective_mask_v1",
            }

            named_checkpoint_policy.record_source_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_role=checkpoint_role,
                context=context,
                metrics=metrics,
                extra=extra_metadata,
            )
            
            # Save telemetry
            if model is not None:
                telemetry = collect_lora_telemetry(model)
                telemetry["step"] = iteration or 0
                telemetry_path = Path(checkpoint_path).parent / "lora_telemetry.json"
                try:
                    telemetry_path.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
                except Exception:
                    pass

            if resume_weights_path is None:
                named_checkpoint_policy.ensure_policy_init(
                    source_checkpoint_path=checkpoint_path,
                    context=context,
                    metrics=metrics,
                )

            if iteration is None:
                return

            named_checkpoint_policy.update_last(
                source_checkpoint_path=checkpoint_path,
                context=context,
                metrics=metrics,
            )
            if latest_val_loss is not None:
                named_checkpoint_policy.update_best_by_eval(
                    source_checkpoint_path=checkpoint_path,
                    metric_name="validation_loss",
                    metric_value=latest_val_loss,
                    higher_is_better=False,
                    context=context,
                    metrics=metrics,
                )
            if _is_epoch_boundary(context):
                named_checkpoint_policy.update_last_epoch_boundary(
                    source_checkpoint_path=checkpoint_path,
                    context=context,
                    metrics=metrics,
                )

        sft_trainer.evaluate = _evaluate_wrapper
        sft_trainer.save_adapter = _save_adapter_wrapper
        loss_fn = partial(
            _vision_language_loss_fn_with_marker_sequences,
            assistant_marker_sequences=assistant_marker_sequences,
        )

        from .adapter_manifest import write_adapter_load_manifest
        manifest_path = Path(plan.args.output_path).expanduser().resolve().parent / "adapter_load_manifest.json"
        write_adapter_load_manifest(
            manifest_path=manifest_path,
            stage=run_id,
            base_model_id=config.model.model_id,
            adapter_path=plan.args.output_path,
            config_path=config.config_path,
            lora_params={
                "lora_rank": config.training.lora_rank,
                "lora_alpha": config.training.lora_alpha,
                "lora_dropout": config.training.lora_dropout,
                "lora_num_layers": getattr(plan.args, "lora_num_layers", None),
            },
            source_resume_adapter=plan.args.adapter_path,
            dataset_path=plan.dataset_path,
            pretokenized_cache_path=config.training.pretokenized_packed_cache_path or config.training.pretokenized_cache_path,
        )
        
        # Append run to registry
        from .run_registry import append_run_record
        import datetime
        append_run_record({
            "run_id": run_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "config_path": str(config.config_path),
            "output_adapter_path": plan.args.output_path
        })

        train_sft_with_milestone_eval(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset_messages,
            args=training_args,
            loss_fn=loss_fn,
            train_on_completions=plan.args.train_on_completions,
            assistant_id=plan.args.assistant_id,
        )
        return plan
    finally:
        if coverage_tracker is not None:
            coverage_tracker.save(force=True)
        if sft_trainer_module is not None:
            if original_iterate_batches is not None:
                sft_trainer_module.iterate_batches = original_iterate_batches
            if original_save_adapter is not None:
                sft_trainer_module.save_adapter = original_save_adapter
            if original_evaluate is not None:
                sft_trainer_module.evaluate = original_evaluate
        if checkpoint_cleanup_stop is not None:
            checkpoint_cleanup_stop.set()
        if checkpoint_cleanup_thread is not None:
            checkpoint_cleanup_thread.join(timeout=5)
            checkpoint_dir = Path(plan.args.output_path).expanduser().resolve().parent
            _prune_stage1_checkpoints(
                checkpoint_dir,
                config.training.checkpoint_keep_last,
                run_id=run_id,
                pin_iterations=frozenset(config.training.checkpoint_pin_iterations),
            )
        if lock_acquired and coverage_tracker is not None:
            _release_run_lock(coverage_tracker.lock_path)


def run_training_smoke_test(
    config: PipelineConfig,
    dataset_path: str | Path | None = None,
    val_dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume_adapter_path: str | Path | None = None,
    run_id: str | None = None,
) -> TrainingPlan:
    plan = plan_training(
        config,
        dataset_path=dataset_path,
        val_dataset_path=val_dataset_path,
        output_path=output_path,
        resume_adapter_path=resume_adapter_path,
        run_id=run_id,
        dry_run=False,
        require_full_opt_in=False,
    )

    if not plan.dataset_path.exists():
        raise RuntimeError(f"Smoke test dataset does not exist: {plan.dataset_path}")
    smoke_iters = config.training.dry_run_steps
    return _execute_training(
        config,
        plan,
        limit_examples=max(smoke_iters, config.memory.batch_size),
        forced_iters=smoke_iters,
    )


def run_training(
    config: PipelineConfig,
    dataset_path: str | Path | None = None,
    val_dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume_adapter_path: str | Path | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    iters: int | None = None,
    save_interval: int = 100,
) -> TrainingPlan:
    plan = plan_training(
        config,
        dataset_path=dataset_path,
        val_dataset_path=val_dataset_path,
        output_path=output_path,
        resume_adapter_path=resume_adapter_path,
        run_id=run_id,
        dry_run=dry_run,
        iters=iters,
        save_interval=save_interval,
    )
    if iters is not None:
        plan.args.iters = iters
        plan.args.epochs = None

    if dry_run:
        return plan
    return _execute_training(config, plan)
