from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from .compiler import CompilerService
from .prepare import (
    DEFAULT_DATASET_ID,
    add_local_figure,
    check_hf_dataset_readiness,
    prepare_hf_dataset,
    split_prepared_dataset,
)
from .promotion import run_sft_promotion_gate
from .refine import RefinementOrchestrator
from .settings import ensure_runtime_directories, load_config
from .train import run_training, run_training_smoke_test
from .train_stage2 import run_stage2_training, run_stage2_training_smoke_test


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _read_description(args: argparse.Namespace) -> str:
    if args.description:
        return args.description
    if args.description_file:
        return Path(args.description_file).read_text(encoding="utf-8")
    raise ValueError("Either --description or --description-file is required.")


def command_validate_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    _print_json(
        {
            "config_path": config.config_path,
            "model_id": config.model.model_id,
            "batch_size": config.memory.batch_size,
            "gradient_checkpointing": config.memory.gradient_checkpointing,
            "freeze_vision": config.memory.freeze_vision,
            "allow_full_training": config.training.allow_full_training,
            "outputs_dir": config.paths.outputs_dir,
        }
    )
    return 0


def command_compile(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    compiler = CompilerService(config.compiler)
    latex_source = Path(args.tex_file).read_text(encoding="utf-8")
    summary = compiler.compile_document(latex_source, output_dir=args.output_dir, job_name="input")
    _print_json(
        {
            "status": summary.status.value,
            "return_code": summary.return_code,
            "key_errors": summary.key_errors,
            "line_hints": summary.line_hints,
            "missing_packages": summary.missing_packages,
            "pdf_path": summary.pdf_path,
            "log_path": summary.log_path,
            "working_dir": summary.working_dir,
        }
    )
    return 0


def command_infer(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    description = _read_description(args)
    orchestrator = RefinementOrchestrator(config)
    result = orchestrator.run(description, args.output_dir)
    _print_json(
        {
            "final_status": result.final_status.value,
            "output_dir": args.output_dir,
            "attempts": [
                {
                    "mode": attempt.mode.value,
                    "status": attempt.compile_summary.status.value if attempt.compile_summary else None,
                    "pdf_path": attempt.compile_summary.pdf_path if attempt.compile_summary else None,
                    "log_path": attempt.compile_summary.log_path if attempt.compile_summary else None,
                    "debug_image_path": attempt.debug_image_path,
                }
                for attempt in result.attempts
            ],
            "final_code": result.final_code,
        }
    )
    return 0


def _run_post_training_ab_eval(config, plan) -> None:
    """Run A/B evaluation after training finishes.

    Discovers intermediate checkpoints for progression tracking,
    then runs base model + finetuned comparison on fixed examples.
    """
    adapter_path = plan.output_path
    if not adapter_path.exists():
        print("[ab_eval] Skipping: final adapter not found.", flush=True)
        return

    ab_eval_script = Path(__file__).resolve().parent.parent.parent / "tools" / "ab_eval.py"
    if not ab_eval_script.exists():
        print(f"[ab_eval] Skipping: {ab_eval_script} not found.", flush=True)
        return

    config_path = config.config_path or "configs/lora_prod.yaml"
    checkpoint_dir = str(adapter_path.parent)

    cmd = [
        sys.executable,
        str(ab_eval_script),
        "--config", str(config_path),
        "--adapter-path", str(adapter_path),
        "--checkpoint-dir", checkpoint_dir,
        "--num-samples", "50",
        "--seed", "42",
        "--max-tokens", "2048",
    ]

    print(f"\n{'═' * 60}", flush=True)
    print("  Running post-training A/B evaluation...", flush=True)
    print(f"  Adapter: {adapter_path}", flush=True)
    print(f"  Checkpoints: {checkpoint_dir}", flush=True)
    print(f"{'═' * 60}\n", flush=True)

    try:
        result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent.parent))
        if result.returncode != 0:
            print(f"[ab_eval] WARNING: evaluation exited with code {result.returncode}", flush=True)
    except Exception as exc:
        print(f"[ab_eval] WARNING: evaluation failed: {exc}", flush=True)


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.smoke_run:
        plan = run_training_smoke_test(
            config,
            dataset_path=args.dataset,
            val_dataset_path=args.val_dataset,
            output_path=args.output_path,
            resume_adapter_path=args.resume_adapter,
            run_id=args.run_id,
        )
    else:
        plan = run_training(
            config,
            dataset_path=args.dataset,
            val_dataset_path=args.val_dataset,
            output_path=args.output_path,
            resume_adapter_path=args.resume_adapter,
            run_id=args.run_id,
            dry_run=args.dry_run,
            iters=getattr(args, "iters", None),
        )
    _print_json(
        {
            "smoke_run": args.smoke_run,
            "dry_run": plan.dry_run,
            "dataset_path": plan.dataset_path,
            "val_dataset_path": plan.val_dataset_path,
            "output_path": plan.output_path,
            "warnings": plan.warnings,
            "args": vars(plan.args),
        }
    )

    # Run A/B eval after training completes unless explicitly skipped.
    if not args.dry_run and not args.smoke_run and not getattr(args, "skip_post_ab_eval", False):
        _run_post_training_ab_eval(config, plan)

    return 0


def command_train_stage2(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.smoke_run:
        plan = run_stage2_training_smoke_test(
            config,
            dataset_path=args.dataset,
            output_path=args.output_path,
            resume_adapter_path=args.resume_adapter,
            run_id=args.run_id,
        )
    else:
        plan = run_stage2_training(
            config,
            dataset_path=args.dataset,
            output_path=args.output_path,
            resume_adapter_path=args.resume_adapter,
            run_id=args.run_id,
            dry_run=args.dry_run,
            iters=getattr(args, "iters", None),
        )
    _print_json(
        {
            "smoke_run": args.smoke_run,
            "dry_run": plan.dry_run,
            "dataset_path": plan.dataset_path,
            "output_path": plan.output_path,
            "checkpoint_dir": plan.checkpoint_dir,
            "reward_cache_dir": plan.reward_cache_dir,
            "warnings": plan.warnings,
            "args": vars(plan.args),
        }
    )
    return 0


def command_promote_sft(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)

    sft_final_path = Path(args.sft_final_path) if args.sft_final_path else (config.paths.runs_dir / "sft_final.safetensors")
    policy_init_path = (
        Path(args.policy_init_path)
        if args.policy_init_path
        else (config.paths.runs_dir / "policy_init.safetensors")
    )

    result = run_sft_promotion_gate(
        baseline_report_path=Path(args.baseline_report),
        candidate_report_path=Path(args.candidate_report),
        baseline_key=args.baseline_key,
        candidate_key=args.candidate_key,
        min_compile_delta=args.min_compile_delta,
        min_schema_delta=args.min_schema_delta,
        min_candidate_compile_rate=args.min_candidate_compile_rate,
        min_candidate_schema_rate=args.min_candidate_schema_rate,
        baseline_compile_rate=args.baseline_compile_rate,
        candidate_compile_rate=args.candidate_compile_rate,
        promote=args.promote,
        candidate_checkpoint_path=Path(args.candidate_checkpoint) if args.candidate_checkpoint else None,
        sft_final_path=sft_final_path,
        policy_init_path=policy_init_path,
        force_policy_init=args.force_policy_init,
        run_id=args.run_id,
        gate_config_path=Path(args.gate_config) if args.gate_config else None,
    )
    _print_json(result)
    return 0


def command_prepare_dataset(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    allowed_sources = set(args.source) if args.source else None
    summary = prepare_hf_dataset(
        config,
        dataset_id=args.dataset_id,
        split=args.split,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        allowed_sources=allowed_sources,
        progress_interval=args.progress_interval,
    )
    _print_json(
        {
            "dataset_id": summary.dataset_id,
            "split": summary.split,
            "total_seen": summary.total_seen,
            "total_written": summary.total_written,
            "total_rejected": summary.total_rejected,
            "total_duplicates": summary.total_duplicates,
            "truncated_records": summary.truncated_records,
            "p99_token_length": summary.p99_token_length,
            "max_context_tokens": summary.max_context_tokens,
            "train_path": summary.train_path,
            "stage2_path": summary.stage2_path,
            "images_dir": summary.images_dir,
            "manifest_path": summary.manifest_path,
            "counts_by_environment": summary.counts_by_environment,
            "counts_by_source": summary.counts_by_source,
            "rejected_reasons": summary.rejected_reasons,
        }
    )
    return 0


def command_harden_dataset(args: argparse.Namespace) -> int:
    from .harden import harden_jsonl_dataset
    config = load_config(args.config)
    cache_path = Path(args.cache) if getattr(args, "cache", None) else None
    summary = harden_jsonl_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        config=config,
        max_violations=args.max_violations,
        timeout_seconds=args.timeout,
        min_description_quality=args.min_description_quality,
        check_bounding_box=not args.no_bbox_check,
        perceptual_dedup=not args.no_perceptual_dedup,
        curriculum_sort=not args.no_curriculum_sort,
        cache_path=cache_path,
        max_workers=getattr(args, "workers", None),
    )
    _print_json(summary)
    return 0 if summary["status"] == "success" else 1


def command_split_dataset(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    summary = split_prepared_dataset(
        config,
        train_path=args.train_path,
        stage2_path=args.stage2_path,
        val_fraction=args.val_fraction,
        gold_eval_fraction=args.gold_eval_fraction,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "source_train_path": summary.source_train_path,
            "source_stage2_path": summary.source_stage2_path,
            "train_path": summary.train_path,
            "val_path": summary.val_path,
            "gold_eval_path": summary.gold_eval_path,
            "train_stage2_path": summary.train_stage2_path,
            "val_stage2_path": summary.val_stage2_path,
            "gold_eval_stage2_path": summary.gold_eval_stage2_path,
            "manifest_path": summary.manifest_path,
            "total_records": summary.total_records,
            "train_records": summary.train_records,
            "val_records": summary.val_records,
            "gold_eval_records": summary.gold_eval_records,
            "train_stage2_records": summary.train_stage2_records,
            "val_stage2_records": summary.val_stage2_records,
            "gold_eval_stage2_records": summary.gold_eval_stage2_records,
            "missing_stage2_records": summary.missing_stage2_records,
            "grouped_keys": summary.grouped_keys,
        }
    )
    return 0


def command_check_dataset(args: argparse.Namespace) -> int:
    summary = check_hf_dataset_readiness(
        dataset_id=args.dataset_id,
        split=args.split,
        sample_limit=args.sample_limit,
    )
    usable_fraction = 0.0
    if summary.checked_records > 0:
        usable_fraction = summary.usable_records / summary.checked_records

    _print_json(
        {
            "dataset_id": summary.dataset_id,
            "split": summary.split,
            "sample_limit": args.sample_limit,
            "checked_records": summary.checked_records,
            "usable_records": summary.usable_records,
            "usable_fraction": usable_fraction,
            "missing_tikz_code": summary.missing_tikz_code,
            "missing_description": summary.missing_description,
            "records_with_images": summary.records_with_images,
            "ready": summary.usable_records > 0 and math.isfinite(usable_fraction),
        }
    )
    return 0


def command_add_figure(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    description = _read_description(args)
    summary = add_local_figure(
        config,
        tex_path=args.tex_file,
        description=description,
        image_path=args.image_file,
        source=args.source,
    )
    _print_json(
        {
            "total_written": summary.total_written,
            "train_path": summary.train_path,
            "stage2_path": summary.stage2_path,
            "images_dir": summary.images_dir,
            "manifest_path": summary.manifest_path,
            "counts_by_environment": summary.counts_by_environment,
            "counts_by_source": summary.counts_by_source,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tikz-mlx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("--config", required=True)
    validate_parser.set_defaults(func=command_validate_config)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--config", required=True)
    compile_parser.add_argument("--tex-file", required=True)
    compile_parser.add_argument("--output-dir", required=True)
    compile_parser.set_defaults(func=command_compile)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--config", required=True)
    description_group = infer_parser.add_mutually_exclusive_group(required=True)
    description_group.add_argument("--description")
    description_group.add_argument("--description-file")
    infer_parser.add_argument("--output-dir", required=True)
    infer_parser.set_defaults(func=command_infer)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--dataset")
    train_parser.add_argument("--val-dataset")
    train_parser.add_argument("--output-path")
    train_parser.add_argument("--resume-adapter")
    train_parser.add_argument("--run-id")
    train_parser.add_argument("--iters", type=int)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--smoke-run", action="store_true")
    train_parser.add_argument(
        "--skip-post-ab-eval",
        action="store_true",
        help="Skip automatic post-training tools/ab_eval.py run for this train invocation.",
    )
    train_parser.set_defaults(func=command_train)

    train_stage2_parser = subparsers.add_parser("train-stage2")
    train_stage2_parser.add_argument("--config", required=True)
    train_stage2_parser.add_argument("--dataset")
    train_stage2_parser.add_argument("--output-path")
    train_stage2_parser.add_argument("--resume-adapter")
    train_stage2_parser.add_argument("--run-id")
    train_stage2_parser.add_argument("--iters", type=int)
    train_stage2_parser.add_argument("--dry-run", action="store_true")
    train_stage2_parser.add_argument("--smoke-run", action="store_true")
    train_stage2_parser.set_defaults(func=command_train_stage2)

    promote_sft_parser = subparsers.add_parser("promote-sft")
    promote_sft_parser.add_argument("--config", required=True)
    promote_sft_parser.add_argument("--baseline-report", required=True)
    promote_sft_parser.add_argument("--candidate-report", required=True)
    promote_sft_parser.add_argument("--baseline-key")
    promote_sft_parser.add_argument("--candidate-key")
    promote_sft_parser.add_argument("--baseline-compile-rate", type=float)
    promote_sft_parser.add_argument("--candidate-compile-rate", type=float)
    promote_sft_parser.add_argument("--min-compile-delta", type=float, default=0.0)
    promote_sft_parser.add_argument("--min-schema-delta", type=float, default=0.0)
    promote_sft_parser.add_argument("--min-candidate-compile-rate", type=float, default=0.0)
    promote_sft_parser.add_argument("--min-candidate-schema-rate", type=float, default=0.0)
    promote_sft_parser.add_argument("--promote", action="store_true")
    promote_sft_parser.add_argument("--candidate-checkpoint")
    promote_sft_parser.add_argument("--sft-final-path")
    promote_sft_parser.add_argument("--policy-init-path")
    promote_sft_parser.add_argument("--force-policy-init", action="store_true")
    promote_sft_parser.add_argument("--gate-config")
    promote_sft_parser.add_argument("--run-id", default="sft_promotion_gate")
    promote_sft_parser.set_defaults(func=command_promote_sft)

    prepare_parser = subparsers.add_parser("prepare-dataset")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    prepare_parser.add_argument("--split", default="train")
    prepare_parser.add_argument("--max-samples", type=int)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--source", action="append")
    prepare_parser.add_argument("--progress-interval", type=int, default=1000)
    prepare_parser.set_defaults(func=command_prepare_dataset)

    harden_parser = subparsers.add_parser(
        "harden-dataset",
        help="Apply critic, repair, dedup, score, and sort a JSONL training dataset.",
    )
    harden_parser.add_argument("--config", required=True)
    harden_parser.add_argument("--input", required=True)
    harden_parser.add_argument("--output", required=True)
    harden_parser.add_argument("--max-violations", type=int, default=0,
        help="Max static critic violations tolerated (default: 0 = hard only).")
    harden_parser.add_argument("--timeout", type=float, default=10.0,
        help="Tectonic compile timeout per sample in seconds (default: 10).")
    harden_parser.add_argument("--min-description-quality", type=float, default=0.3,
        help="Minimum description quality score 0–1 (default: 0.3).")
    harden_parser.add_argument("--cache", default=None,
        help="Path to SQLite compiler cache (enables skip-recompile on reruns).")
    harden_parser.add_argument("--no-bbox-check", action="store_true",
        help="Disable bounding-box check for invisible diagrams.")
    harden_parser.add_argument("--no-perceptual-dedup", action="store_true",
        help="Disable perceptual hash deduplication.")
    harden_parser.add_argument("--no-curriculum-sort", action="store_true",
        help="Skip sorting by pedagogical score (curriculum ordering).")
    harden_parser.add_argument("--workers", type=int, default=None,
        help="Number of parallel compile threads. Defaults to 75%% of CPU cores.")
    harden_parser.set_defaults(func=command_harden_dataset)

    check_dataset_parser = subparsers.add_parser("check-dataset")
    check_dataset_parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    check_dataset_parser.add_argument("--split", default="train")
    check_dataset_parser.add_argument("--sample-limit", type=int, default=128)
    check_dataset_parser.set_defaults(func=command_check_dataset)

    split_parser = subparsers.add_parser("split-dataset")
    split_parser.add_argument("--config", required=True)
    split_parser.description = (
        "Split immutable prepared source datasets into train/val/gold outputs. "
        "Avoid using output files as split sources to prevent collapse of val/gold splits."
    )
    split_parser.add_argument("--train-path")
    split_parser.add_argument("--stage2-path")
    split_parser.add_argument("--val-fraction", type=float, default=0.1)
    split_parser.add_argument("--gold-eval-fraction", type=float, default=0.05)
    split_parser.add_argument("--overwrite", action="store_true")
    split_parser.set_defaults(func=command_split_dataset)

    add_figure_parser = subparsers.add_parser("add-figure")
    add_figure_parser.add_argument("--config", required=True)
    add_figure_parser.add_argument("--tex-file", required=True)
    add_figure_parser.add_argument("--image-file")
    add_figure_parser.add_argument("--source", default="local")
    add_description_group = add_figure_parser.add_mutually_exclusive_group(required=True)
    add_description_group.add_argument("--description")
    add_description_group.add_argument("--description-file")
    add_figure_parser.set_defaults(func=command_add_figure)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
