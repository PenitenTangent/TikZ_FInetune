import argparse
import json
import os
import shutil
import socket
from pathlib import Path

import numpy as np
import pytest
import yaml

import tikz_mlx.train as train_module
from tikz_mlx.adapter_config_io import validate_resumed_adapter_lora_hyperparams
from tikz_mlx.checkpointing import checkpoint_metadata_path
from tikz_mlx.curriculum_diagnostics import (
    iter_jsonl_metadata_token_lengths,
    summarize_token_lengths,
)
from tikz_mlx.lr_schedule_probe import build_sft_joined_lr_schedule, lr_value_at_step
from tikz_mlx.dataset import (
    build_epoch_example_order,
    compute_example_order_checksum,
    validate_row_aligned_example_indices,
)
from tikz_mlx.mlx_runtime import MlxRuntimeUnavailableError, ensure_mlx_runtime_available
from tikz_mlx.settings import load_config
from tikz_mlx.train import (
    EnhancedVisionDataset,
    PackedPreTokenizedDataset,
    StrictCoverageTracker,
    TrainingPlan,
    _acquire_run_lock,
    _build_assistant_marker_sequences,
    _build_syntax_weight_lookup,
    _build_strict_iterate_batches,
    _capacity_upgrade_resume_fingerprints,
    _build_training_batch,
    _compute_assistant_response_indices,
    _compute_mask_zero_fraction,
    _compute_training_config_fingerprint,
    _dataset_loader_spec,
    _load_and_validate_pack_audit,
    _prune_stage1_checkpoints,
    _read_checkpoint_iteration,
    _read_checkpoint_resume_info,
    _repetition_unlikelihood_loss,
    _release_run_lock,
    _resolve_training_iterations,
    _vision_language_loss_fn_with_marker_sequences,
    build_lora_namespace,
    collect_lora_targets,
    is_allowed_lora_target_name,
    plan_training,
    run_training,
)
from tikz_mlx.train_stage2 import (
    _should_persist_rollout_artifacts,
    _update_rollout_retention_queue,
)
from tools.build_curriculum import (
    allocate_stage_iters,
    filter_val_jsonl_by_min_metadata_tokens,
    load_train_records,
    require_stage3_pretokenize_long_coverage,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lora_prod.yaml"
CLEAN_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "clean_adapter_recovery.yaml"


def _skip_if_mlx_unavailable() -> None:
    try:
        ensure_mlx_runtime_available()
    except MlxRuntimeUnavailableError as exc:
        pytest.skip(str(exc))


def test_dataset_loader_spec_uses_json_builder_for_jsonl() -> None:
    dataset_name, data_files = _dataset_loader_spec(Path("data/prepared/train.jsonl"))
    assert dataset_name == "json"
    assert data_files == {"train": "data/prepared/train.jsonl"}


def test_build_lora_namespace_uses_json_loader_metadata_for_local_jsonl() -> None:
    config = load_config(CONFIG_PATH)
    args = build_lora_namespace(
        config,
        Path("data/prepared/train.jsonl"),
        None,
        Path("runs/tikz_lora_adapter.safetensors"),
        dry_run=False,
        run_id="test-run",
    )
    assert args.dataset == "json"
    assert args.data_files == {"train": "data/prepared/train.jsonl"}
    assert args.epochs == config.training.epochs
    assert args.steps_per_save == config.training.steps_per_save
    assert args.grad_clip == 1.0


def test_build_lora_namespace_wires_validation_split_for_local_jsonl() -> None:
    config = load_config(CONFIG_PATH)
    args = build_lora_namespace(
        config,
        Path("data/prepared/train.jsonl"),
        Path("data/prepared/val.jsonl"),
        Path("runs/tikz_lora_adapter.safetensors"),
        dry_run=False,
        run_id="test-run",
    )
    assert args.data_files["train"] == "data/prepared/train.jsonl"
    assert args.data_files["validation"] == "data/prepared/val.jsonl"
    assert args.val_split == "validation"


def test_load_config_exposes_stage2_defaults() -> None:
    config = load_config(CONFIG_PATH)
    assert config.training.resume_adapter_path is None
    assert config.training.steps_per_save == 2000
    assert config.training.steps_per_eval == 200
    assert config.training.checkpoint_keep_last == 1
    assert config.training.checkpoint_cleanup_interval_seconds == 20
    assert config.training.auto_resume_latest_checkpoint is False
    assert config.memory.gradient_accumulation_steps == 4
    assert config.memory.retry_cache_policy == "clear"
    assert config.training.val_dataset_path is not None
    assert config.training.gold_eval_dataset_path is not None
    assert config.training.require_nonempty_validation_dataset is True
    assert config.training.require_nonempty_gold_eval_dataset is True
    assert config.training.min_validation_compilation_rate == 0.0
    assert config.training.validation_compile_probe_limit == 25
    assert config.training.coverage.enabled is False
    assert config.training.coverage.strict_fingerprint is True
    assert config.training.coverage.require_state_for_resume is True
    assert config.training.coverage.save_interval_steps == 200
    assert config.training.coverage.order_mode == "epoch-shuffle"
    assert config.training.coverage.order_seed_base == 17
    assert config.training.completion_mask_preflight_enabled is True
    assert config.training.completion_mask_preflight_rows == 2500
    assert config.training.completion_mask_preflight_min_marker_hit_rate == pytest.approx(0.9)
    assert config.training.completion_mask_preflight_min_mask_zero_fraction == pytest.approx(0.01)
    assert config.training.reward_weighted_loss is False
    assert config.training.reward_weight_field == "sample_weight"
    assert config.training.syntax_weighted_loss is True
    assert config.training.repetition_unlikelihood_enabled is True
    assert config.training.repetition_unlikelihood_weight == pytest.approx(0.05)
    assert config.training.repetition_unlikelihood_window == 64
    assert config.training.repetition_unlikelihood_min_context == 16
    assert config.training.syntax_structural_weight == pytest.approx(5.0)
    assert config.training.syntax_command_weight == pytest.approx(2.0)
    assert config.training.syntax_coordinate_weight == pytest.approx(1.0)
    assert config.training.stage2.reward_backend == "emd"
    assert config.training.stage2.rollout_group_size == 8
    assert config.training.stage2.iters == 100
    assert config.training.stage2.val_dataset_path is not None
    assert config.training.stage2.gold_eval_dataset_path is not None
    assert config.training.stage2.reward_compile_floor == 0.655
    assert config.training.stage2.promotion_min_compile_rate == 0.815
    assert config.training.stage2.dead_signal_watchdog_enabled is True
    assert config.training.stage2.dead_signal_watchdog_windows == 3
    assert config.training.stage2.dead_signal_watchdog_min_format_reject_rate == pytest.approx(0.95)
    assert config.training.stage2.dead_signal_watchdog_min_truncated_rate == pytest.approx(0.95)
    assert config.training.stage2.rollout_debug_save_every == 1
    assert config.training.stage2.rollout_debug_max_dirs == 0
    assert config.training.stage2.rollout_debug_force_final_save is True


def test_load_config_rejects_invalid_repetition_unlikelihood_values(tmp_path: Path) -> None:
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "  repetition_unlikelihood_weight: 0.05\n",
        "  repetition_unlikelihood_weight: -0.01\n",
    )
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="repetition_unlikelihood_weight"):
        load_config(bad_config)


def test_load_config_exposes_inference_candidate_and_decoding_profile() -> None:
    config = load_config(CONFIG_PATH)
    assert config.inference.initial_candidates == 4
    assert config.inference.compile_repair_candidates == 2
    assert config.inference.visual_repair_candidates == 2
    assert config.inference.initial_decoding.max_tokens == 2048
    assert config.inference.initial_decoding.min_p == 0.05
    assert config.inference.compile_repair_decoding.temperature == 0.3
    assert config.inference.visual_repair_decoding.repetition_penalty == 1.06


def test_clean_adapter_config_uses_plain_ce_and_staged_lr() -> None:
    config = load_config(CLEAN_CONFIG_PATH)

    assert config.training.train_dataset_path.as_posix().endswith("data/prepared/train_unified.jsonl")
    assert config.training.learning_rate == pytest.approx(4e-6)
    assert config.training.weight_decay == pytest.approx(0.05)
    assert config.training.lora_num_layers == 28
    assert config.training.lora_rank == 16
    assert config.training.lora_alpha == 32
    assert config.training.lora_dropout == pytest.approx(0.03)
    assert config.training.resume_adapter_path is None
    assert config.training.auto_resume_latest_checkpoint is False
    assert config.training.reward_weighted_loss is False
    assert config.training.syntax_weighted_loss is False
    assert config.training.stage2.enabled is False
    assert config.training.pretokenized_packed_cache_path is None
    assert config.training.coverage.enabled is True
    assert config.training.coverage.save_interval_steps == 50
    assert config.training.checkpoint_pin_iterations == (50_000, 100_000, 150_000)


def test_curriculum_stage0_uses_current_stable_params_without_unlikelihood() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "curriculum_stage0.yaml"
    config = load_config(config_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config.memory.gradient_accumulation_steps == 8
    assert raw_config["training"]["iters"] == 1600
    assert raw_config["training"]["iters"] % config.memory.gradient_accumulation_steps == 0
    assert config.training.learning_rate == pytest.approx(1.0e-6)
    assert config.training.lr_warmup_fraction == pytest.approx(0.05)
    assert config.training.lora_rank == 16
    assert config.training.lora_alpha == 32
    assert config.training.lora_num_layers == 42
    assert config.training.lora_dropout == pytest.approx(0.08)
    assert config.model.max_context_tokens == 768
    assert config.training.repetition_unlikelihood_enabled is False
    assert config.training.repetition_unlikelihood_weight == pytest.approx(0.0)
    assert config.training.collapse_probe.enabled is True
    assert config.training.collapse_probe.interval_steps == 50
    assert config.training.val_batches == 2
    assert config.training.validation_compile_probe_limit == 2
    assert config.inference.initial_decoding.repetition_penalty == pytest.approx(1.3)
    assert config.inference.initial_decoding.no_repeat_ngram_size == 4


def test_plan_training_threads_resume_adapter_into_namespace() -> None:
    config = load_config(CONFIG_PATH)
    resume_path = Path("runs/resume_adapter.safetensors").resolve()
    plan = plan_training(config, resume_adapter_path=resume_path, dry_run=True)
    assert plan.args.adapter_path == str(resume_path)
    assert any("does not exist yet" in warning for warning in plan.warnings)


def test_should_persist_rollout_artifacts_respects_save_every_and_final_override() -> None:
    assert _should_persist_rollout_artifacts(
        global_step=4,
        local_iteration=4,
        total_iters=10,
        save_every=2,
        force_final_save=False,
    )
    assert not _should_persist_rollout_artifacts(
        global_step=5,
        local_iteration=5,
        total_iters=10,
        save_every=2,
        force_final_save=False,
    )
    assert _should_persist_rollout_artifacts(
        global_step=9,
        local_iteration=10,
        total_iters=10,
        save_every=100,
        force_final_save=True,
    )


def test_update_rollout_retention_queue_refreshes_duplicate_path_without_self_prune(tmp_path: Path) -> None:
    queue: dict[Path, int] = {}
    first = tmp_path / "iter_0000010"
    first.mkdir(parents=True, exist_ok=True)
    stale = _update_rollout_retention_queue(queue, output_root=first, global_step=10, max_kept=1)
    assert stale == []
    stale = _update_rollout_retention_queue(queue, output_root=first, global_step=20, max_kept=1)
    assert stale == []
    assert list(queue.keys()) == [first]
    assert queue[first] == 20


def test_update_rollout_retention_queue_prunes_oldest_when_max_dirs_one(tmp_path: Path) -> None:
    queue: dict[Path, int] = {}
    first = tmp_path / "iter_0000010"
    second = tmp_path / "iter_0000020"
    stale = _update_rollout_retention_queue(queue, output_root=first, global_step=10, max_kept=1)
    assert stale == []
    stale = _update_rollout_retention_queue(queue, output_root=second, global_step=20, max_kept=1)
    assert stale == [first]
    assert list(queue.keys()) == [second]


def test_allocate_stage_iters_preserves_exact_budget() -> None:
    out = allocate_stage_iters([100, 30, 5], total_iters=1500, n_records=135)
    assert sum(out) == 1500
    assert len(out) == 3


def test_load_train_records_respects_max_examples(tmp_path: Path) -> None:
    p = tmp_path / "train.jsonl"
    rows = [{"metadata": {"token_length": 100 + i}} for i in range(20)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    recs, tls = load_train_records(p, max_examples=5)
    assert len(recs) == 5
    assert tls == [100, 101, 102, 103, 104]


def test_load_train_records_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "train.jsonl"
    p.write_text(
        "\n\n" + json.dumps({"metadata": {"token_length": 7}}) + "\n",
        encoding="utf-8",
    )
    recs, tls = load_train_records(p)
    assert len(recs) == 1
    assert tls == [7]


def test_build_lora_namespace_uses_config_val_batches_default() -> None:
    config = load_config(CONFIG_PATH)
    args = build_lora_namespace(
        config,
        Path("data/prepared/train.jsonl"),
        None,
        Path("runs/tikz_lora_adapter.safetensors"),
        dry_run=False,
        run_id="test-run",
    )
    assert args.val_batches == config.training.val_batches == 25


def test_sft_lr_schedule_reaches_cosine_floor_at_final_step() -> None:
    _skip_if_mlx_unavailable()
    import mlx.optimizers as optim

    peak = 3e-5
    for total in (730, 541, 228):
        sched = build_sft_joined_lr_schedule(
            optim, peak_lr=peak, total_steps=total, warmup_fraction=0.1, cosine_end_fraction=0.01
        )
        end_lr = peak * 0.01
        assert lr_value_at_step(sched, total) == pytest.approx(end_lr, rel=0.0, abs=1e-12)
        assert lr_value_at_step(sched, total + 1) == pytest.approx(end_lr, rel=0.0, abs=1e-12)


def test_validate_resumed_adapter_rejects_rank_mismatch(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "ckpt"
    adapter_dir.mkdir()
    (adapter_dir / "adapters.safetensors").write_bytes(b"")
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"rank": 4, "alpha": 16, "dropout": 0.1}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="LoRA hyperparameters"):
        validate_resumed_adapter_lora_hyperparams(
            adapter_path=adapter_dir,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )


def test_iter_jsonl_metadata_token_lengths_and_val_warn_profile(tmp_path: Path) -> None:
    val_file = tmp_path / "val.jsonl"
    rows = [
        {"metadata": {"token_length": 400}},
        {"metadata": {"token_length": 500}},
        {"metadata": {"token_length": 2000}},
    ]
    val_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    lengths = iter_jsonl_metadata_token_lengths(val_file)
    assert lengths == [400, 500, 2000]
    summary = summarize_token_lengths(lengths)
    assert summary["count"] == 3
    assert summary["max"] == 2000


def test_require_stage3_pretokenize_long_coverage_passes_and_fails(tmp_path: Path) -> None:
    ok_audit = tmp_path / "good_audit.json"
    ok_audit.write_text(
        json.dumps({"kept_length_ge_fractions": {"1024": 0.5}}),
        encoding="utf-8",
    )
    require_stage3_pretokenize_long_coverage(audit_path=ok_audit, min_fraction_ge_1024=0.02)

    require_stage3_pretokenize_long_coverage(audit_path=ok_audit, min_fraction_ge_1024=0.0)

    bad_audit = tmp_path / "bad_audit.json"
    bad_audit.write_text(
        json.dumps({"kept_length_ge_fractions": {"1024": 0.001}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="long-context coverage"):
        require_stage3_pretokenize_long_coverage(audit_path=bad_audit, min_fraction_ge_1024=0.02)


def test_curriculum_stage_configs_use_lora_num_layers_28_and_strict_coverage() -> None:
    repo = Path(__file__).resolve().parents[1]
    for name in (
        "curriculum_stage1.yaml",
        "curriculum_stage2.yaml",
        "curriculum_stage3.yaml",
        "curriculum_stage4.yaml",
        "curriculum_stage5.yaml",
    ):
        path = repo / "configs" / name
        if not path.exists():
            continue
        cfg = load_config(path)
        assert cfg.training.lora_num_layers >= 28
        assert cfg.training.coverage.enabled is True
        assert cfg.training.repetition_unlikelihood_enabled is True
        assert 0.01 <= cfg.training.repetition_unlikelihood_weight <= 0.10
        assert cfg.training.repetition_unlikelihood_window == 64
        assert cfg.training.repetition_unlikelihood_min_context == 16
        assert cfg.training.repetition_unlikelihood_warmup_steps == 500


def test_curriculum_stage2_switches_to_1024_after_70_percent() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "curriculum_stage2.yaml")

    assert cfg.model.max_context_tokens == 1024
    assert cfg.training.max_seq_length_schedule == ((0.0, 768), (0.7, 1024))
    assert cfg.training.repetition_unlikelihood_weight == pytest.approx(0.02)


def test_long_context_curriculum_stages_use_safer_schedules() -> None:
    repo = Path(__file__).resolve().parents[1]
    stage3 = load_config(repo / "configs" / "curriculum_stage3.yaml")
    stage4 = load_config(repo / "configs" / "curriculum_stage4.yaml")
    stage5 = load_config(repo / "configs" / "curriculum_stage5.yaml")

    assert stage3.training.learning_rate == pytest.approx(1.0e-6)
    assert stage3.training.lora_dropout == pytest.approx(0.06)
    assert stage3.training.repetition_unlikelihood_weight == pytest.approx(0.03)
    assert stage3.training.max_seq_length_schedule == ((0.0, 1024), (0.4, 1280), (0.75, 1536))

    assert stage4.training.learning_rate == pytest.approx(7.5e-7)
    assert stage4.training.lora_dropout == pytest.approx(0.05)
    assert stage4.training.repetition_unlikelihood_weight == pytest.approx(0.03)
    assert stage4.training.max_seq_length_schedule == ((0.0, 1536), (0.6, 1792))

    assert stage5.training.learning_rate == pytest.approx(5.0e-7)
    assert stage5.training.lora_dropout == pytest.approx(0.04)
    assert stage5.training.repetition_unlikelihood_weight == pytest.approx(0.02)


def test_collect_lora_targets_reports_expected_suffixes() -> None:
    class FakeLoRaLayer:
        pass

    class FakeLanguageModel:
        def __init__(self) -> None:
            self.layers = {
                "model.layers.0.self_attn.q_proj": FakeLoRaLayer(),
                "model.layers.0.self_attn.k_proj": FakeLoRaLayer(),
                "model.layers.0.self_attn.v_proj": FakeLoRaLayer(),
                "model.layers.0.self_attn.o_proj": FakeLoRaLayer(),
                "model.layers.0.mlp.gate_proj": FakeLoRaLayer(),
                "model.layers.0.mlp.up_proj": FakeLoRaLayer(),
                "model.layers.0.mlp.down_proj": FakeLoRaLayer(),
            }

        def named_modules(self):
            return list(self.layers.items())

    class FakeModel:
        language_model = FakeLanguageModel()

    audit = collect_lora_targets(FakeModel())

    assert audit["target_count"] == 7
    assert audit["missing_expected_suffixes"] == []
    assert all(audit["expected_suffix_hits"].values())
    assert audit["suffix_counts"]["q_proj"] == 1
    assert audit["layer_indices"] == [0]
    assert audit["unexpected_targets"] == []


def test_collect_lora_targets_reports_undercovered_suffixes() -> None:
    class FakeLoRaLayer:
        pass

    class FakeLanguageModel:
        def named_modules(self):
            return [
                ("model.layers.6.self_attn.q_proj", FakeLoRaLayer()),
                ("model.layers.7.self_attn.q_proj", FakeLoRaLayer()),
                ("model.layers.6.self_attn.v_proj", FakeLoRaLayer()),
            ]

    class FakeModel:
        language_model = FakeLanguageModel()

    audit = collect_lora_targets(
        FakeModel(),
        expected_lora_num_layers=2,
        expected_min_layer=6,
        expected_max_layer=7,
    )

    assert audit["suffix_counts"]["q_proj"] == 2
    assert audit["undercovered_suffixes"]["v_proj"] == 1
    assert audit["missing_expected_suffixes"]
    assert audit["unexpected_layer_indices_below_min"] == []


def test_collect_lora_targets_reports_unexpected_non_whitelisted_targets() -> None:
    class FakeLoRaLayer:
        pass

    class FakeLanguageModel:
        def named_modules(self):
            return [
                ("model.layers.0.self_attn.q_proj", FakeLoRaLayer()),
                ("model.layers.0.per_layer_projection", FakeLoRaLayer()),
                ("model.per_layer_model_projection", FakeLoRaLayer()),
            ]

    class FakeModel:
        language_model = FakeLanguageModel()

    audit = collect_lora_targets(
        FakeModel(),
        expected_lora_num_layers=1,
        expected_min_layer=0,
        expected_max_layer=0,
    )

    assert is_allowed_lora_target_name(
        "language_model.model.layers.0.self_attn.q_proj",
        cutoff=0,
        total_layers=1,
    )
    assert audit["unexpected_targets"] == [
        "language_model.model.layers.0.per_layer_projection",
        "language_model.model.per_layer_model_projection",
    ]


def test_strict_coverage_rejects_filename_resume_offset() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "curriculum_stage1.yaml")

    with pytest.raises(RuntimeError, match="filename-derived resume offsets"):
        run_training(cfg, dry_run=True, resume_offset=1000)


def test_filter_val_jsonl_by_min_metadata_tokens_keeps_long_rows(tmp_path: Path) -> None:
    src = tmp_path / "val.jsonl"
    rows = [
        {"sample_id": "a", "metadata": {"token_length": 100}},
        {"sample_id": "b", "metadata": {"token_length": 900}},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "val_slice.jsonl"
    kept, scanned = filter_val_jsonl_by_min_metadata_tokens(
        val_src=src, val_out=out, min_metadata_tokens=500
    )
    assert scanned == 2
    assert kept == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["sample_id"] == "b"


def test_allocate_stage_iters_handles_budget_smaller_than_non_empty_stage_count() -> None:
    out = allocate_stage_iters([10, 10, 10], total_iters=2, n_records=30)
    assert sum(out) == 2
    assert out.count(1) == 2


def test_build_syntax_weight_lookup_classifies_structural_command_and_coordinate() -> None:
    class _Tokenizer:
        def get_vocab(self):
            return {"{": 0, r"\draw": 1, "12.5": 2, "node": 3}

        def decode(self, ids):
            inverse = {0: "{", 1: r"\draw", 2: "12.5", 3: "node"}
            return inverse[ids[0]]

    lookup = _build_syntax_weight_lookup(
        _Tokenizer(),
        structural_weight=5.0,
        command_weight=2.0,
        coordinate_weight=1.0,
    )
    assert lookup[0] == pytest.approx(5.0)
    assert lookup[1] == pytest.approx(2.0)
    assert lookup[2] == pytest.approx(1.0)
    assert lookup[3] == pytest.approx(1.0)


def test_packed_pretokenized_dataset_can_derive_syntax_weights_from_lookup() -> None:
    _skip_if_mlx_unavailable()
    packed = PackedPreTokenizedDataset(
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        np.array([[1, -1]], dtype=np.int32),
        np.array([[0, 0, 1, 1]], dtype=np.uint8),
        syntax_weight_lookup=np.array([5.0, 2.0, 1.0, 1.0], dtype=np.float16),
    )
    item = packed[0]
    assert np.array(item["syntax_weight"]).tolist() == pytest.approx([5.0, 2.0, 1.0, 1.0])


def test_load_and_validate_pack_audit_rejects_missing_audit(tmp_path: Path) -> None:
    packed_path = tmp_path / "train_packed.npy"
    masks_path = tmp_path / "train_packed_masks.npy"
    boundaries_path = tmp_path / "train_packed_boundaries.npy"
    np.save(packed_path, np.zeros((1, 8), dtype=np.int32))
    np.save(masks_path, np.zeros((1, 8), dtype=np.uint8))
    np.save(boundaries_path, np.zeros((1, 2), dtype=np.int32))

    with pytest.raises(RuntimeError, match="No pack audit found"):
        _load_and_validate_pack_audit(
            packed_path=packed_path,
            assistant_id=4368,
            min_marker_hit_rate=0.9,
            min_mask_zero_fraction=0.01,
        )


def test_load_and_validate_pack_audit_accepts_matching_artifacts(tmp_path: Path) -> None:
    packed_path = tmp_path / "train_packed.npy"
    masks_path = tmp_path / "train_packed_masks.npy"
    boundaries_path = tmp_path / "train_packed_boundaries.npy"
    audit_path = tmp_path / "train_packed_audit.json"
    np.save(packed_path, np.zeros((1, 8), dtype=np.int32))
    np.save(masks_path, np.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.uint8))
    np.save(boundaries_path, np.array([[2, -1]], dtype=np.int32))

    import hashlib

    def _digest(*paths: Path) -> str:
        hasher = hashlib.sha256()
        for path in paths:
            hasher.update(path.name.encode("utf-8"))
            hasher.update(path.read_bytes())
        return hasher.hexdigest()

    audit_path.write_text(
        json.dumps(
            {
                "marker_hit_rate": 1.0,
                "mask_zero_fraction": 0.75,
                "assistant_token_used": 4368,
                "dataset_sha256": _digest(packed_path, masks_path, boundaries_path),
            }
        ),
        encoding="utf-8",
    )

    payload = _load_and_validate_pack_audit(
        packed_path=packed_path,
        assistant_id=4368,
        min_marker_hit_rate=0.9,
        min_mask_zero_fraction=0.01,
    )
    assert payload["marker_hit_rate"] == 1.0


def test_load_and_validate_pack_audit_rejects_metadata_jsonl_for_plain_ce(tmp_path: Path) -> None:
    packed_path = tmp_path / "train_packed.npy"
    masks_path = tmp_path / "train_packed_masks.npy"
    boundaries_path = tmp_path / "train_packed_boundaries.npy"
    audit_path = tmp_path / "train_packed_audit.json"
    np.save(packed_path, np.zeros((1, 8), dtype=np.int32))
    np.save(masks_path, np.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.uint8))
    np.save(boundaries_path, np.array([[2, -1]], dtype=np.int32))

    import hashlib

    def _digest(*paths: Path) -> str:
        hasher = hashlib.sha256()
        for path in paths:
            hasher.update(path.name.encode("utf-8"))
            hasher.update(path.read_bytes())
        return hasher.hexdigest()

    audit_path.write_text(
        json.dumps(
            {
                "marker_hit_rate": 1.0,
                "mask_zero_fraction": 0.75,
                "assistant_token_used": 4368,
                "dataset_sha256": _digest(packed_path, masks_path, boundaries_path),
                "metadata_jsonl": str(tmp_path / "train_scored.jsonl"),
                "scoring_status": "metadata_weighted",
                "reward_weighted": False,
                "syntax_weighted": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="metadata_jsonl"):
        _load_and_validate_pack_audit(
            packed_path=packed_path,
            assistant_id=4368,
            min_marker_hit_rate=0.9,
            min_mask_zero_fraction=0.01,
        )


def test_plan_training_assigns_default_run_id_from_output_stem() -> None:
    config = load_config(CONFIG_PATH)
    plan = plan_training(
        config,
        output_path=Path("runs/custom_adapter.safetensors"),
        dry_run=True,
    )
    assert plan.args.run_id == "custom_adapter"


def test_plan_training_respects_explicit_run_id() -> None:
    config = load_config(CONFIG_PATH)
    plan = plan_training(
        config,
        output_path=Path("runs/custom_adapter.safetensors"),
        run_id="explicit-run",
        dry_run=True,
    )
    assert plan.args.run_id == "explicit-run"


def test_plan_training_converts_safetensors_resume_file_to_adapter_directory(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    config.paths.runs_dir = runs_dir

    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"messages": []}\n', encoding="utf-8")

    checkpoint = runs_dir / "0000100_adapters.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    (runs_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    plan = plan_training(
        config,
        dataset_path=dataset_path,
        resume_adapter_path=checkpoint,
        dry_run=True,
        require_full_opt_in=False,
    )

    adapter_path = Path(plan.args.adapter_path)
    assert adapter_path.is_dir()
    assert (adapter_path / "adapter_config.json").exists()
    assert (adapter_path / "adapters.safetensors").exists()


def test_plan_training_auto_resumes_latest_checkpoint(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    config.paths.runs_dir = runs_dir
    config.training.auto_resume_latest_checkpoint = True
    config.training.resume_adapter_path = None
    config.training.require_nonempty_validation_dataset = False
    config.training.require_nonempty_gold_eval_dataset = False

    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"messages": []}\n', encoding="utf-8")

    oldest = runs_dir / "0000100_adapters.safetensors"
    newest = runs_dir / "0000200_adapters.safetensors"
    oldest.write_bytes(b"old")
    newest.write_bytes(b"new")
    checkpoint_metadata_path(oldest).write_text('{"run_id": "tikz_lora_adapter"}\n', encoding="utf-8")
    checkpoint_metadata_path(newest).write_text('{"run_id": "tikz_lora_adapter"}\n', encoding="utf-8")
    (runs_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    os.utime(oldest, (1, 1))
    os.utime(newest, (2, 2))

    plan = plan_training(
        config,
        dataset_path=dataset_path,
        dry_run=False,
        require_full_opt_in=False,
    )

    adapter_path = Path(plan.args.adapter_path)
    assert adapter_path.is_dir()
    assert (adapter_path / "adapters.safetensors").read_bytes() == b"new"
    assert any("Auto-resuming from latest checkpoint" in warning for warning in plan.warnings)


def test_training_config_fingerprint_ignores_config_path(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"messages": []}\n', encoding="utf-8")

    plan = plan_training(
        config,
        dataset_path=dataset_path,
        dry_run=True,
        require_full_opt_in=False,
    )
    first = _compute_training_config_fingerprint(config, plan)

    config.config_path = tmp_path / "alternate_config.yaml"
    second = _compute_training_config_fingerprint(config, plan)

    assert first == second


def test_training_config_fingerprint_tracks_loss_and_schedule_settings(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"messages": []}\n', encoding="utf-8")
    plan = plan_training(
        config,
        dataset_path=dataset_path,
        dry_run=True,
        require_full_opt_in=False,
    )

    first = _compute_training_config_fingerprint(config, plan)
    config.training.repetition_unlikelihood_weight += 0.01
    second = _compute_training_config_fingerprint(config, plan)
    config.training.repetition_unlikelihood_weight -= 0.01
    config.training.max_seq_length_schedule = ((0.0, 512), (0.5, 768))
    third = _compute_training_config_fingerprint(config, plan)

    assert first != second
    assert first != third


def test_prune_stage1_checkpoints_scoped_to_run_id(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step, run_id in ((1, "run-a"), (2, "run-a"), (3, "run-a"), (4, "run-b")):
        checkpoint = checkpoint_dir / f"{step:07d}_adapters.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_metadata_path(checkpoint).write_text(
            json.dumps({"run_id": run_id}),
            encoding="utf-8",
        )

    deleted = _prune_stage1_checkpoints(checkpoint_dir, keep_last=1, run_id="run-a")

    assert [path.name for path in deleted] == [
        "0000001_adapters.safetensors",
        "0000002_adapters.safetensors",
    ]
    assert not (checkpoint_dir / "0000001_adapters.safetensors").exists()
    assert not (checkpoint_dir / "0000002_adapters.safetensors").exists()
    assert (checkpoint_dir / "0000003_adapters.safetensors").exists()
    assert (checkpoint_dir / "0000004_adapters.safetensors").exists()
    assert not checkpoint_metadata_path(checkpoint_dir / "0000001_adapters.safetensors").exists()
    assert not checkpoint_metadata_path(checkpoint_dir / "0000002_adapters.safetensors").exists()


def test_prune_stage1_checkpoints_keeps_pin_iterations_even_when_stale(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rid = "run-a"
    base_ts = 1_700_000_000
    for idx, step in enumerate((10_000, 50_000, 51_000, 52_000, 200_000)):
        checkpoint = checkpoint_dir / f"{step:07d}_adapters.safetensors"
        checkpoint.write_bytes(b"x")
        checkpoint_metadata_path(checkpoint).write_text(
            json.dumps({"run_id": rid}),
            encoding="utf-8",
        )
        ts = base_ts + idx
        os.utime(checkpoint, (ts, ts))

    deleted = _prune_stage1_checkpoints(
        checkpoint_dir,
        keep_last=2,
        run_id=rid,
        pin_iterations=frozenset({50_000, 100_000, 150_000}),
    )
    names = {path.name for path in deleted}
    assert "0050000_adapters.safetensors" not in names
    assert (checkpoint_dir / "0050000_adapters.safetensors").exists()
    assert names == {"0010000_adapters.safetensors", "0051000_adapters.safetensors"}


def test_acquire_run_lock_reclaims_stale_local_lock(tmp_path: Path) -> None:
    run_id = "unit-run"
    lock_path = tmp_path / "train.lock"
    lock_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pid": 99999999,
                "host": socket.gethostname(),
                "acquired_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    _acquire_run_lock(lock_path, run_id)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert int(payload["pid"]) == os.getpid()

    _release_run_lock(lock_path)
    assert not lock_path.exists()


def test_resolve_training_iterations_prefers_epochs_for_full_training() -> None:
    args = argparse.Namespace(batch_size=1, epochs=2, iters=0)
    assert _resolve_training_iterations(3, args) == 6


def test_resolve_training_iterations_uses_iters_without_epochs() -> None:
    args = argparse.Namespace(batch_size=1, epochs=None, iters=4)
    assert _resolve_training_iterations(3, args) == 4


def test_plan_training_rejects_non_dry_missing_validation_when_required(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text('{"messages": []}\n', encoding="utf-8")
    config.training.require_nonempty_validation_dataset = True
    config.training.require_nonempty_gold_eval_dataset = False
    config.training.val_dataset_path = tmp_path / "missing_val.jsonl"

    with pytest.raises(RuntimeError, match="Validation dataset path does not exist"):
        plan_training(
            config,
            dataset_path=dataset_path,
            dry_run=False,
            require_full_opt_in=False,
        )


def test_plan_training_rejects_non_dry_empty_gold_when_required(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dataset_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    dataset_path.write_text('{"messages": []}\n', encoding="utf-8")
    val_path.write_text('{"messages": []}\n', encoding="utf-8")
    gold_path.write_text("", encoding="utf-8")
    config.training.require_nonempty_validation_dataset = True
    config.training.require_nonempty_gold_eval_dataset = True
    config.training.val_dataset_path = val_path
    config.training.gold_eval_dataset_path = gold_path

    with pytest.raises(RuntimeError, match="Gold-eval dataset is empty"):
        plan_training(
            config,
            dataset_path=dataset_path,
            dry_run=False,
            require_full_opt_in=False,
        )


def test_validate_row_aligned_example_indices_rejects_non_aligned_sequence() -> None:
    with pytest.raises(ValueError):
        validate_row_aligned_example_indices([1, 0])


def test_compute_assistant_response_indices_prefers_marker_sequence_over_assistant_id() -> None:
    input_ids = np.array([[1, 77091, 2, 101, 102, 103, 4]], dtype=np.int32)
    result = _compute_assistant_response_indices(
        input_ids,
        assistant_id=77091,
        marker_sequences=((101, 102, 103),),
    )
    assert result.tolist() == [5]


def test_compute_assistant_response_indices_falls_back_to_assistant_id() -> None:
    input_ids = np.array([[1, 77091, 2, 3]], dtype=np.int32)
    result = _compute_assistant_response_indices(
        input_ids,
        assistant_id=77091,
        marker_sequences=((101, 102, 103),),
    )
    assert result.tolist() == [1]


def test_compute_mask_zero_fraction_matches_expected_shifted_mask_behavior() -> None:
    attention_mask = np.array([1, 1, 1, 1], dtype=np.int32)
    zero_fraction = _compute_mask_zero_fraction(attention_mask, assistant_boundary_index=1)
    # Boundary at index 1 zeroes positions [0, 1], and shifted weights use positions [1, 2, 3].
    assert zero_fraction == pytest.approx(1 / 3)


def test_build_training_batch_handles_leading_batch_dimension() -> None:
    _skip_if_mlx_unavailable()
    item = {
        "input_ids": np.array([[11, 12, 13, 14]], dtype=np.int32),
        "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int32),
        "pixel_values": None,
    }
    batch = _build_training_batch([item], max_seq_length=16)
    input_ids = np.array(batch["input_ids"])
    attention_mask = np.array(batch["attention_mask"])

    assert input_ids.shape[0] == 1
    assert input_ids.shape[1] >= 4
    assert input_ids[0, :4].tolist() == [11, 12, 13, 14]
    assert attention_mask[0, :4].tolist() == [1, 1, 1, 1]


def test_repetition_unlikelihood_penalizes_recent_non_target_probability() -> None:
    _skip_if_mlx_unavailable()
    import mlx.core as mx

    logits_clean = mx.zeros((1, 12, 8), dtype=mx.float32)
    logits_repetitive = mx.array(np.zeros((1, 12, 8), dtype=np.float32))
    # Prefix [1, 2, 3] has already occurred twice. At the final position,
    # token 4 would continue a repeated 4-gram loop, while gold token 6 breaks it.
    logits_repetitive = logits_repetitive.at[0, 11, 4].add(8.0)
    labels = mx.array([[1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 6]], dtype=mx.int32)
    mask = mx.ones((1, 12), dtype=mx.float32)

    clean = _repetition_unlikelihood_loss(
        logits=logits_clean,
        labels=labels,
        effective_mask=mask,
        window=12,
        min_context=0,
    )
    repetitive = _repetition_unlikelihood_loss(
        logits=logits_repetitive,
        labels=labels,
        effective_mask=mask,
        window=12,
        min_context=0,
    )

    mx.eval(clean, repetitive)
    assert repetitive.item() > clean.item()


def test_repetition_unlikelihood_excludes_current_gold_repetition() -> None:
    _skip_if_mlx_unavailable()
    import mlx.core as mx

    logits = mx.array(np.zeros((1, 4, 8), dtype=np.float32))
    logits = logits.at[0, 3, 2].add(8.0)
    labels = mx.array([[2, 2, 2, 2]], dtype=mx.int32)
    mask = mx.ones((1, 4), dtype=mx.float32)

    loss = _repetition_unlikelihood_loss(
        logits=logits,
        labels=labels,
        effective_mask=mask,
        window=4,
        min_context=0,
    )

    mx.eval(loss)
    assert loss.item() < 0.01


def test_repetition_unlikelihood_ignores_non_loop_repetition() -> None:
    _skip_if_mlx_unavailable()
    import mlx.core as mx

    logits = mx.array(np.zeros((1, 6, 8), dtype=np.float32))
    logits = logits.at[0, 5, 2].add(8.0)
    labels = mx.array([[1, 2, 3, 2, 4, 5]], dtype=mx.int32)
    mask = mx.ones((1, 6), dtype=mx.float32)

    loss = _repetition_unlikelihood_loss(
        logits=logits,
        labels=labels,
        effective_mask=mask,
        window=4,
        min_context=0,
    )

    mx.eval(loss)
    assert loss.item() < 0.01


def test_vision_language_loss_repetition_aux_can_be_disabled_and_masked() -> None:
    _skip_if_mlx_unavailable()
    import mlx.core as mx

    class _Output:
        def __init__(self, logits):
            self.logits = logits

    class _FakeModel:
        def __init__(self, logits):
            self.logits = logits

        def __call__(self, input_ids, pixel_values, attention_mask, **kwargs):
            return _Output(self.logits)

    logits = mx.array(np.zeros((1, 5, 8), dtype=np.float32))
    logits = logits.at[0, 2, 2].add(8.0)
    batch = {
        "input_ids": mx.array([[0, 1, 2, 3, 2, 4]], dtype=mx.int32),
        "attention_mask": mx.ones((1, 6), dtype=mx.int32),
        "pixel_values": mx.zeros((1, 1), dtype=mx.float32),
        # Shifted mask trains no label positions here, so repeated-token
        # probabilities in prompt-only positions must not affect the loss.
        "weight_mask": mx.array([[0, 0, 0, 0, 0, 0]], dtype=mx.float32),
    }
    model = _FakeModel(logits)

    disabled = _vision_language_loss_fn_with_marker_sequences(
        model,
        batch,
        train_on_completions=True,
        repetition_unlikelihood_enabled=False,
        repetition_unlikelihood_weight=0.05,
        repetition_unlikelihood_window=4,
        repetition_unlikelihood_min_context=0,
    )
    enabled_masked = _vision_language_loss_fn_with_marker_sequences(
        model,
        batch,
        train_on_completions=True,
        repetition_unlikelihood_enabled=True,
        repetition_unlikelihood_weight=0.05,
        repetition_unlikelihood_window=4,
        repetition_unlikelihood_min_context=0,
    )
    zero_weight = _vision_language_loss_fn_with_marker_sequences(
        model,
        batch,
        train_on_completions=True,
        repetition_unlikelihood_enabled=True,
        repetition_unlikelihood_weight=0.0,
        repetition_unlikelihood_window=4,
        repetition_unlikelihood_min_context=0,
    )

    mx.eval(disabled, enabled_masked, zero_weight)
    assert enabled_masked.item() == pytest.approx(disabled.item())
    assert zero_weight.item() == pytest.approx(disabled.item())


def test_build_assistant_marker_sequences_uses_tokenizer_encode_without_special_tokens() -> None:
    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True):  # pragma: no cover - signature parity
            if text == "<start_of_turn>model\n":
                return [11, 12, 13]
            if text == "<|start_of_turn|>model\n":
                return [11, 12, 13]
            if text == "<start_of_turn>assistant\n":
                return [21, 22]
            if text == "<|start_of_turn|>assistant\n":
                return [21, 22]
            return []

    markers = _build_assistant_marker_sequences(_FakeTokenizer())
    assert markers == ((11, 12, 13), (21, 22))


def test_epoch_order_is_deterministic_and_has_stable_checksum() -> None:
    order_a = build_epoch_example_order(
        total_examples=8,
        order_mode="epoch-shuffle",
        seed_base=17,
        epoch=3,
    )
    order_b = build_epoch_example_order(
        total_examples=8,
        order_mode="epoch-shuffle",
        seed_base=17,
        epoch=3,
    )

    assert order_a == order_b
    assert compute_example_order_checksum(order_a) == compute_example_order_checksum(order_b)


def test_strict_coverage_tracker_initializes_and_updates_state(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config.paths.runs_dir = tmp_path / "runs"

    tracker = StrictCoverageTracker(
        config=config,
        run_id="unit-run",
        run_dir=config.paths.runs_dir / "unit-run",
        dataset_fingerprint={"dataset_path": "x", "line_count": 4, "sha256": "abc"},
        config_fingerprint="cfg",
        total_examples=4,
        target_steps=8,
        resume_requested=False,
    )

    first_example = tracker.peek_next_example_index()
    tracker.mark_batch_complete(first_example)
    assert tracker.state.global_step == 1
    assert tracker.remaining_steps == 7
    assert tracker.state.next_example_index in {0, 1, 2, 3}
    assert tracker.state_path.exists()


def test_strict_coverage_tracker_syncs_to_checkpoint_step(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config.paths.runs_dir = tmp_path / "runs"

    tracker = StrictCoverageTracker(
        config=config,
        run_id="unit-run",
        run_dir=config.paths.runs_dir / "unit-run",
        dataset_fingerprint={"dataset_path": "x", "line_count": 4, "sha256": "abc"},
        config_fingerprint="cfg",
        total_examples=4,
        target_steps=8,
        resume_requested=False,
    )

    tracker.sync_to_global_step(5)
    assert tracker.state.global_step == 5
    assert tracker.state.epoch == 1
    assert tracker.state.batch_cursor_in_epoch == 1
    assert tracker.remaining_steps == 3

    tracker.sync_to_global_step(2)
    assert tracker.state.global_step == 2
    assert tracker.state.epoch == 0
    assert tracker.state.batch_cursor_in_epoch == 2
    assert tracker.remaining_steps == 6


def test_read_checkpoint_iteration_uses_canonical_sidecar_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "0002560_adapters.safetensors"
    checkpoint.write_bytes(b"weights")
    checkpoint_metadata_path(checkpoint).write_text(
        json.dumps({"global_step": 3072, "run_id": "curriculum_stage1"}),
        encoding="utf-8",
    )

    info = _read_checkpoint_resume_info(checkpoint)
    assert info is not None
    assert info.global_step == 3072
    assert info.run_id == "curriculum_stage1"
    assert _read_checkpoint_iteration(checkpoint) == 3072


def test_read_checkpoint_iteration_falls_back_to_numeric_checkpoint_name(tmp_path: Path) -> None:
    checkpoint = tmp_path / "0002560_adapters.safetensors"
    checkpoint.write_bytes(b"weights")

    info = _read_checkpoint_resume_info(checkpoint)
    assert info is not None
    assert info.global_step == 2560
    assert info.run_id is None


def test_coverage_resume_request_treats_different_run_as_warm_start(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last_probe_pass_adapters.safetensors"
    checkpoint.write_bytes(b"weights")
    checkpoint_metadata_path(checkpoint).write_text(
        json.dumps({"global_step": 246, "run_id": "curriculum_stage0"}),
        encoding="utf-8",
    )

    resume_requested, info = train_module._coverage_resume_request(
        checkpoint,
        "curriculum_stage1",
    )

    assert resume_requested is False
    assert info is not None
    assert info.run_id == "curriculum_stage0"
    assert info.global_step == 246


def test_coverage_resume_request_keeps_same_run_as_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last_probe_pass_adapters.safetensors"
    checkpoint.write_bytes(b"weights")
    checkpoint_metadata_path(checkpoint).write_text(
        json.dumps({"global_step": 512, "run_id": "curriculum_stage1"}),
        encoding="utf-8",
    )

    resume_requested, info = train_module._coverage_resume_request(
        checkpoint,
        "curriculum_stage1",
    )

    assert resume_requested is True
    assert info is not None
    assert info.global_step == 512


def test_materialized_numeric_resume_checkpoint_records_current_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    config.paths.runs_dir = tmp_path / "runs"
    checkpoint = tmp_path / "0002560_adapters.safetensors"
    checkpoint.write_bytes(b"weights")

    def _fake_materialize_lora_handoff_adapter(**kwargs):
        target_dir = kwargs["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "adapters.safetensors").write_bytes(b"weights")
        return {
            "source_rank": kwargs["target_rank"],
            "source_dropout": kwargs["target_dropout"],
            "expanded": False,
            "alpha_rescaled": False,
        }

    monkeypatch.setattr(
        train_module,
        "materialize_lora_handoff_adapter",
        _fake_materialize_lora_handoff_adapter,
    )

    adapter_dir = train_module._prepare_resume_adapter_directory(
        config,
        checkpoint,
        warnings=[],
        run_id="curriculum_stage1",
    )

    payload = json.loads((adapter_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert payload["global_step"] == 2560
    assert payload["run_id"] == "curriculum_stage1"


def test_strict_coverage_accepts_lora_capacity_upgrade_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    config.paths.runs_dir = tmp_path / "runs"
    config.training.lora_rank = 24
    config.training.lora_alpha = 48
    config.training.lora_dropout = 0.08
    config.training.lora_num_layers = 42

    dataset_path = tmp_path / "train.jsonl"
    checkpoint = tmp_path / "0002560_adapters.safetensors"
    checkpoint.write_bytes(b"weights")
    checkpoint_metadata_path(checkpoint).write_text(
        json.dumps(
            {
                "global_step": 4,
                "run_id": "unit-run",
                "resolved_training_config": {
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.08,
                },
            }
        ),
        encoding="utf-8",
    )

    def _fake_materialize_lora_handoff_adapter(**kwargs):
        target_dir = kwargs["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "adapters.safetensors").write_bytes(b"weights")
        return {
            "source_rank": 16,
            "source_alpha": 32,
            "source_dropout": 0.08,
            "target_rank": kwargs["target_rank"],
            "target_alpha": kwargs["target_alpha"],
            "target_dropout": kwargs["target_dropout"],
            "expanded": True,
            "alpha_rescaled": False,
        }

    monkeypatch.setattr(
        train_module,
        "materialize_lora_handoff_adapter",
        _fake_materialize_lora_handoff_adapter,
    )
    adapter_dir = train_module._prepare_resume_adapter_directory(
        config,
        checkpoint,
        warnings=[],
        run_id="unit-run",
    )

    args = argparse.Namespace(
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        epochs=None,
        steps_per_save=512,
        train_on_completions=True,
        lora_rank=24,
        lora_alpha=48,
        lora_dropout=0.08,
        lora_num_layers=42,
        adapter_path=str(adapter_dir),
    )
    plan = TrainingPlan(
        dataset_path=dataset_path,
        val_dataset_path=None,
        output_path=tmp_path / "out.safetensors",
        dry_run=False,
        args=args,
        warnings=[],
    )
    current_fingerprint = _compute_training_config_fingerprint(config, plan)
    source_fingerprint = _compute_training_config_fingerprint(
        config,
        plan,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.08,
    )
    accepted = _capacity_upgrade_resume_fingerprints(config, plan, current_fingerprint)
    assert source_fingerprint in accepted

    run_dir = config.paths.runs_dir / "unit-run"
    StrictCoverageTracker(
        config=config,
        run_id="unit-run",
        run_dir=run_dir,
        dataset_fingerprint={"dataset_path": "x", "line_count": 4, "sha256": "abc"},
        config_fingerprint=source_fingerprint,
        total_examples=4,
        target_steps=8,
        resume_requested=False,
    )

    tracker = StrictCoverageTracker(
        config=config,
        run_id="unit-run",
        run_dir=run_dir,
        dataset_fingerprint={"dataset_path": "x", "line_count": 4, "sha256": "abc"},
        config_fingerprint=current_fingerprint,
        total_examples=4,
        target_steps=8,
        resume_requested=True,
        accepted_config_fingerprints=accepted,
    )
    assert tracker.state.config_fingerprint == current_fingerprint


def test_strict_coverage_tracker_recreates_missing_state_parent(tmp_path: Path) -> None:
    config = load_config(CLEAN_CONFIG_PATH)
    config.paths.runs_dir = tmp_path / "runs"
    run_dir = config.paths.runs_dir / "unit-run"

    tracker = StrictCoverageTracker(
        config=config,
        run_id="unit-run",
        run_dir=run_dir,
        dataset_fingerprint={"dataset_path": "x", "line_count": 2, "sha256": "abc"},
        config_fingerprint="cfg",
        total_examples=2,
        target_steps=2,
        resume_requested=False,
    )

    shutil.rmtree(run_dir)
    tracker.save(force=True)

    assert tracker.state_path.exists()


def test_strict_coverage_tracker_rejects_fingerprint_mismatch_on_resume(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config.paths.runs_dir = tmp_path / "runs"
    run_dir = config.paths.runs_dir / "unit-run"

    StrictCoverageTracker(
        config=config,
        run_id="unit-run",
        run_dir=run_dir,
        dataset_fingerprint={"dataset_path": "x", "line_count": 2, "sha256": "abc"},
        config_fingerprint="cfg",
        total_examples=2,
        target_steps=2,
        resume_requested=False,
    )

    with pytest.raises(RuntimeError, match="Dataset fingerprint mismatch"):
        StrictCoverageTracker(
            config=config,
            run_id="unit-run",
            run_dir=run_dir,
            dataset_fingerprint={"dataset_path": "x", "line_count": 2, "sha256": "different"},
            config_fingerprint="cfg",
            total_examples=2,
            target_steps=2,
            resume_requested=True,
        )


def test_strict_coverage_iterator_does_not_mark_unacknowledged_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Tracker:
        marked: list[int] = []

        def peek_next_example_index(self) -> int:
            return 0

        def mark_batch_complete(self, example_index: int) -> None:
            self.marked.append(example_index)

    monkeypatch.setattr(
        train_module,
        "_build_training_batch",
        lambda items, max_seq_length: {"input_ids": [1, 2, 3]},
    )
    tracker = _Tracker()
    iterator_factory = _build_strict_iterate_batches(
        tracker=tracker,
        original_iterate_batches=lambda *args, **kwargs: iter(()),
    )

    iterator = iterator_factory([{"input_ids": [1, 2, 3]}], batch_size=1, max_seq_length=16, train=True)
    batch = next(iterator)
    assert batch["__tikz_example_index__"] == 0

    iterator.close()
    assert tracker.marked == []
