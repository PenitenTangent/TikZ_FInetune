from __future__ import annotations

import math
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import mlx_vlm.trainer.sft_trainer as sft_trainer


def train(
    model: Any,
    optimizer: Any,
    train_dataset: Any,
    val_dataset: Any = None,
    args: Any = sft_trainer.TrainingArgs(),
    loss_fn: Any = sft_trainer.vision_language_loss_fn,
    train_on_completions: bool = False,
    assistant_id: int = 77091,
    processor: Any = None,  # For probe
) -> None:
    """Run mlx-vlm SFT with optional explicit validation milestones.

    mlx-vlm's built-in trainer evaluates at step 1, every ``steps_per_eval``,
    and final step. The TikZ curriculum pipeline can set ``args._tikz_eval_at``
    to a set of 1-based iteration numbers; when present, those milestones
    become the validation schedule while preserving the upstream training loop.
    """
    mx = sft_trainer.mx
    nn = sft_trainer.nn
    import mlx.utils as mx_utils
    from .collapse_probe import run_collapse_probe_suite
    from .prompting import build_generation_prompt

    if not hasattr(args, "_tikz_eval_at"):
        sft_trainer.train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            loss_fn=loss_fn,
            train_on_completions=train_on_completions,
            assistant_id=assistant_id,
        )
        return

    if mx.metal.is_available():
        device_info = mx.device_info()
        max_working_set_size = device_info.get("max_recommended_working_set_size")
        if max_working_set_size is not None:
            mx.set_wired_limit(max_working_set_size)
    print(f"{sft_trainer.Colors.HEADER}Starting training..., scheduled batches: {args.iters}{sft_trainer.Colors.ENDC}")

    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if val_dataset is None and rank == 0:
        print(f"{sft_trainer.Colors.OKBLUE}No validation dataset provided - training will run without validation.{sft_trainer.Colors.ENDC}")

    if args.grad_checkpoint:
        for module in model.children().values():
            if hasattr(module, "layers"):
                sft_trainer.grad_checkpoint(module.layers[0])

    grad_accum_steps = args.gradient_accumulation_steps
    if grad_accum_steps < 1 and args:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.batch_size % world_size != 0:
        raise ValueError(
            "batch_size must be divisible by distributed world size "
            f"(batch_size={args.batch_size}, world_size={world_size})"
        )
    if args.steps_per_save % grad_accum_steps != 0:
        raise ValueError(
            "steps_per_save must land on a gradient accumulation boundary "
            f"(steps_per_save={args.steps_per_save}, "
            f"gradient_accumulation_steps={grad_accum_steps})"
        )
    strict_global_offset = getattr(args, "_tikz_global_step_offset", None)
    strict_global_offset = int(strict_global_offset) if strict_global_offset is not None else None
    total_target_iters = getattr(args, "_tikz_total_target_iters", None)
    total_target_iters = int(total_target_iters) if total_target_iters is not None else None
    resume_offset = int(getattr(args, "resume_offset", 0) or 0)
    local_total_iters = args.iters if strict_global_offset is not None else max(args.iters - resume_offset, 0)
    if local_total_iters % grad_accum_steps != 0:
        raise ValueError(
            "scheduled iterations must land on a gradient accumulation boundary "
            f"(scheduled_iters={local_total_iters}, "
            f"gradient_accumulation_steps={grad_accum_steps})"
        )

    loss_fn_partial = partial(
        loss_fn,
        train_on_completions=train_on_completions,
        assistant_id=assistant_id,
    )
    loss_value_and_grad = nn.value_and_grad(model, loss_fn_partial)
    state = [model.state, optimizer.state, mx.random.state]
    eval_milestones = frozenset(int(x) for x in getattr(args, "_tikz_eval_at", frozenset()))

    raw_seq_schedule = getattr(args, "_tikz_max_seq_length_schedule", ()) or ()
    max_seq_schedule: list[tuple[int, int]] = []
    if raw_seq_schedule:
        schedule_total = int(total_target_iters or args.iters)
        for fraction, max_len in raw_seq_schedule:
            start_step = int(math.floor(float(fraction) * float(schedule_total)))
            max_seq_schedule.append((start_step, int(max_len)))
        max_seq_schedule.sort(key=lambda item: item[0])

    def _scheduled_max_seq_length(global_step: int) -> int:
        configured = int(args.max_seq_length)
        if not max_seq_schedule:
            return configured
        selected = configured
        for start_step, max_len in max_seq_schedule:
            if global_step >= start_step:
                selected = int(max_len)
            else:
                break
        return min(selected, configured)

    def _truncate_batch_sequences(batch: Any, max_len: int) -> Any:
        if max_len <= 0 or not isinstance(batch, dict):
            return batch
        for key, value in list(batch.items()):
            if key in {"pixel_values", "boundary_positions", "__tikz_example_index__"}:
                continue
            if not hasattr(value, "ndim") or not hasattr(value, "shape"):
                continue
            if int(getattr(value, "ndim", 0)) != 2:
                continue
            if value.shape[1] <= max_len:
                continue
            batch[key] = value[:, :max_len]
        return batch

    collapse_probe_enabled = bool(getattr(args, "_tikz_collapse_probe_enabled", True))
    collapse_probe_interval_steps = int(getattr(args, "_tikz_collapse_probe_interval_steps", 500))
    collapse_probe_max_failures = int(getattr(args, "_tikz_collapse_probe_max_failures", 1))
    collapse_probe_save_checkpoint_on_pass = bool(
        getattr(args, "_tikz_collapse_probe_save_checkpoint_on_pass", True)
    )
    on_collapse_probe_pass = getattr(args, "_tikz_on_collapse_probe_pass", None)
    if collapse_probe_interval_steps <= 0:
        collapse_probe_interval_steps = 500
    collapse_probe_interval_steps = max(collapse_probe_interval_steps, grad_accum_steps)
    collapse_probe_interval_steps = int(
        math.ceil(collapse_probe_interval_steps / grad_accum_steps) * grad_accum_steps
    )
    if collapse_probe_max_failures <= 0:
        collapse_probe_max_failures = 1

    def step(
        batch: Any,
        prev_grad: Any,
        do_update: bool,
        *,
        global_step: int,
    ) -> tuple[Any, Any, Any, Any, Any, Any, bool]:
        if "attention_mask" in batch:
            lengths = batch["attention_mask"].sum(axis=1)
        else:
            lengths = mx.full(
                (batch["input_ids"].shape[0],),
                batch["input_ids"].shape[1],
            )

        toks = lengths.sum()

        try:
            lvalue, grad = loss_value_and_grad(model, batch, global_step=global_step)
        except TypeError:
            lvalue, grad = loss_value_and_grad(model, batch)

        if prev_grad is not None:
            grad = mx_utils.tree_map(lambda x, y: x + y, grad, prev_grad)

        did_update = False
        grad_norm = mx.array(0.0)
        clip_scale = mx.array(1.0)
        clipped = mx.array(0.0)
        if do_update:
            grad = sft_trainer.average_gradients(grad)
            if grad_accum_steps > 1:
                grad = mx_utils.tree_map(lambda x: x / grad_accum_steps, grad)

            flat_grad = mx.concatenate([v.reshape(-1) for _, v in mx_utils.tree_flatten(grad)])
            grad_norm = mx.linalg.norm(flat_grad)

            if args.grad_clip is not None:
                clip_scale = mx.minimum(args.grad_clip / mx.maximum(grad_norm, 1e-8), 1.0)
                clipped = mx.where(clip_scale < 0.999999, 1.0, 0.0)
                grad = mx_utils.tree_map(lambda g: g * clip_scale, grad)
            optimizer.update(model, grad)
            grad = None
            did_update = True

        return lvalue, toks, grad, grad_norm, clip_scale, clipped, did_update

    model.train()
    losses = mx.array(0.0)
    n_tokens = mx.array(0.0)
    steps = 0
    update_steps = 0
    trained_tokens = 0
    train_time = 0.0
    grad_accum = None
    grad_norms = mx.array(0.0)
    clip_scales = mx.array(0.0)
    clipped_steps = mx.array(0.0)
    probe_fail_count = 0
    adapter_path = Path(args.adapter_file)
    last_probe_pass_alias = adapter_path.parent / "last_probe_pass_adapters.safetensors"
    last_probe_pass_checkpoint: Path | None = (
        last_probe_pass_alias if last_probe_pass_alias.exists() else None
    )
    if rank == 0:
        adapter_path.parent.mkdir(parents=True, exist_ok=True)

    if strict_global_offset is not None:
        global_step_offset = strict_global_offset
        target_text = f" / {total_target_iters}" if total_target_iters is not None else ""
        if global_step_offset > 0:
            print(
                f"{sft_trainer.Colors.OKBLUE}"
                f"Resuming from global iteration {global_step_offset}{target_text}. "
                f"Running {local_total_iters} remaining batches."
                f"{sft_trainer.Colors.ENDC}"
            )
        iteration_source = range(1, args.iters + 1)
        skip_batches = 0
    else:
        global_step_offset = resume_offset
        if resume_offset > 0:
            print(f"{sft_trainer.Colors.OKBLUE}Resuming from iteration {resume_offset}. Skipping {resume_offset} batches...{sft_trainer.Colors.ENDC}")
        iteration_source = range(resume_offset + 1, args.iters + 1)
        skip_batches = resume_offset
    if global_step_offset % grad_accum_steps != 0:
        raise ValueError(
            "resume offset must land on a gradient accumulation boundary "
            f"(offset={global_step_offset}, gradient_accumulation_steps={grad_accum_steps})"
        )

    import numpy as np

    # CRITICAL: Force deterministic shuffling so resume_offset skips the exact same samples
    # that were processed in the previous run.
    np.random.seed(42)

    batch_iterator = sft_trainer.iterate_batches(
        dataset=train_dataset,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        train=True,
    )

    # Skip batches already processed in previous runs for legacy filename-offset resume.
    for _ in range(skip_batches):
        try:
            next(batch_iterator)
        except StopIteration:
            break

    for it, batch in zip(
        iteration_source,
        batch_iterator,
    ):
        if strict_global_offset is not None:
            local_it = it
            global_it = global_step_offset + local_it
            eval_it = local_it
        else:
            global_it = it
            local_it = it - resume_offset
            eval_it = global_it
        tic = time.perf_counter()

        if val_dataset is not None and eval_it in eval_milestones:
            tic_val = time.perf_counter()
            val_loss = sft_trainer.evaluate(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn_partial,
                train_on_completions=train_on_completions,
                assistant_id=assistant_id,
            )
            model.train()
            val_time = time.perf_counter() - tic_val
            if rank == 0:
                print(
                    f"{sft_trainer.Colors.OKCYAN}Iter {global_it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s{sft_trainer.Colors.ENDC}",
                    flush=True,
                )
            tic = time.perf_counter()

        coverage_example_index = None
        if isinstance(batch, dict):
            coverage_example_index = batch.pop("__tikz_example_index__", None)

        batch = _truncate_batch_sequences(batch, _scheduled_max_seq_length(global_it))

        lvalue, toks, grad_accum, g_norm, clip_scale, clipped, did_update = step(
            batch,
            grad_accum,
            global_it % grad_accum_steps == 0,
            global_step=global_it,
        )
        mx.clear_cache()
        losses += lvalue
        n_tokens += toks
        steps += 1
        if did_update:
            update_steps += 1
            grad_norms += g_norm
            clip_scales += clip_scale
            clipped_steps += clipped
        mx.eval(state, losses, n_tokens, grad_accum, grad_norms, clip_scales, clipped_steps)
        mark_batch_complete = getattr(args, "_tikz_mark_batch_complete", None)
        if coverage_example_index is not None and callable(mark_batch_complete):
            mark_batch_complete(int(coverage_example_index))
        train_time += time.perf_counter() - tic

        if local_it % args.steps_per_report == 0 or local_it == local_total_iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            update_denominator = max(update_steps, 1)
            avg_grad_norm = mx.distributed.all_sum(grad_norms, stream=mx.cpu).item() / (
                update_denominator * world_size
            )
            avg_clip_scale = mx.distributed.all_sum(clip_scales, stream=mx.cpu).item() / (
                update_denominator * world_size
            )
            clipped_step_rate = mx.distributed.all_sum(clipped_steps, stream=mx.cpu).item() / (
                update_denominator * world_size
            )
            n_tokens_total = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = (
                optimizer.learning_rate.item()
                if hasattr(optimizer.learning_rate, "item")
                else args.learning_rate
            )
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens_total) / train_time
            trained_tokens += n_tokens_total
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                print(
                    f"Iter {global_it}: Train loss {sft_trainer.Colors.OKGREEN}{train_loss:.8f}{sft_trainer.Colors.ENDC}, "
                    f"Grad Norm {avg_grad_norm:.4f}, "
                    f"Clip Scale {avg_clip_scale:.3f}, "
                    f"Clipped {clipped_step_rate:.1%}, "
                    f"LR {learning_rate:.3e}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )
                telemetry_path = Path(
                    getattr(
                        args,
                        "_tikz_gradient_telemetry_path",
                        Path(args.adapter_file).parent / "gradient_clip_telemetry.jsonl",
                    )
                )
                telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                with telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "iteration": global_it,
                                "local_iteration": local_it,
                                "train_loss": train_loss,
                                "avg_grad_norm": avg_grad_norm,
                                "max_grad_norm": args.grad_clip,
                                "avg_clip_scale": avg_clip_scale,
                                "clipped_step_rate": clipped_step_rate,
                                "learning_rate": learning_rate,
                                "tokens_per_sec": tokens_sec,
                                "peak_memory_gb": peak_mem,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

            losses = mx.array(0.0)
            n_tokens = mx.array(0.0)
            steps = 0
            update_steps = 0
            grad_norms = mx.array(0.0)
            clip_scales = mx.array(0.0)
            clipped_steps = mx.array(0.0)
            train_time = 0.0

        if global_it % args.steps_per_save == 0 and rank == 0:
            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            sft_trainer.save_adapter(model, adapter_path)
            checkpoint = adapter_path.parent / f"{global_it:07d}_adapters.safetensors"
            sft_trainer.save_adapter(model, checkpoint)
            print(
                f"{sft_trainer.Colors.OKBLUE}Iter {global_it}: Saved adapter to {checkpoint}.{sft_trainer.Colors.ENDC}",
                flush=True,
            )

        if (
            collapse_probe_enabled
            and processor is not None
            and rank == 0
            and global_it % collapse_probe_interval_steps == 0
        ):
            print(
                f"{sft_trainer.Colors.OKBLUE}Iter {global_it}: Running collapse probe...{sft_trainer.Colors.ENDC}",
                flush=True,
            )
            probe_payload = run_collapse_probe_suite(model, processor, build_generation_prompt)
            passed = bool(probe_payload.get("passed"))
            failures = probe_payload.get("failures", [])
            raw_warning = probe_payload.get("raw_greedy_warning", {})
            probe_record = {
                "iteration": int(global_it),
                "local_iteration": int(local_it),
                **probe_payload,
            }
            try:
                with (adapter_path.parent / "collapse_probe_results.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(probe_record, sort_keys=True) + "\n")
            except Exception:
                pass
            if raw_warning and not raw_warning.get("passed", True):
                raw_failures = raw_warning.get("failures", [])
                print(
                    f"{sft_trainer.Colors.WARNING}Raw-greedy collapse warning: "
                    f"{len(raw_failures)} sentinel prompt(s) failed; production probe still controls rollback."
                    f"{sft_trainer.Colors.ENDC}",
                    flush=True,
                )
            if passed:
                probe_fail_count = 0
                if collapse_probe_save_checkpoint_on_pass:
                    # Keep a stable root-level checkpoint for run_stage.sh recovery/resume.
                    # sft_trainer.save_adapter may be wrapped by train.py; let that wrapper
                    # write canonical metadata instead of replacing it here.
                    last_probe_pass_checkpoint = last_probe_pass_alias
                    sft_trainer.save_adapter(model, last_probe_pass_checkpoint)
                    if callable(on_collapse_probe_pass):
                        on_collapse_probe_pass(str(last_probe_pass_checkpoint), int(global_it))
                    print(
                        f"{sft_trainer.Colors.OKGREEN}✓ Collapse probe passed. Updated {last_probe_pass_checkpoint}.{sft_trainer.Colors.ENDC}",
                        flush=True,
                    )
                else:
                    print(f"{sft_trainer.Colors.OKGREEN}✓ Collapse probe passed.{sft_trainer.Colors.ENDC}")
            else:
                probe_fail_count += 1
                print(
                    f"{sft_trainer.Colors.FAIL}Collapse probe failed ({probe_fail_count}/{collapse_probe_max_failures}).{sft_trainer.Colors.ENDC}",
                    flush=True,
                )
                for f in failures:
                    print(f"  - {f['prompt']}: {', '.join(f['reasons'])}")

                report_path = adapter_path.parent / "collapse_probe_failure.json"
                report = {
                    "iteration": int(global_it),
                    "local_iteration": int(local_it),
                    "probe_interval_steps": int(collapse_probe_interval_steps),
                    "max_failures": int(collapse_probe_max_failures),
                    "failures": failures,
                    "raw_greedy_warning": raw_warning,
                    "last_probe_pass_checkpoint": str(last_probe_pass_checkpoint) if last_probe_pass_checkpoint else None,
                    "last_probe_pass_alias": str(last_probe_pass_alias),
                    "suggested_next": {
                        "resume_from": (
                            str(last_probe_pass_checkpoint)
                            if last_probe_pass_checkpoint is not None
                            else None
                        ),
                        "tighten_grad_clip_factor": 0.5,
                        "halve_learning_rate_factor": 0.5,
                    },
                }
                try:
                    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                except Exception:
                    pass

                if probe_fail_count >= collapse_probe_max_failures:
                    print(
                        f"{sft_trainer.Colors.FAIL}FATAL: Collapse probe failed. Rolling back to last passing checkpoint is required; aborting training.{sft_trainer.Colors.ENDC}",
                        flush=True,
                    )
                    if last_probe_pass_checkpoint is not None:
                        print(
                            f"Resume from: {last_probe_pass_checkpoint}",
                            flush=True,
                        )
                    sys.exit(2)

    if rank == 0:
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        sft_trainer.save_adapter(model, adapter_path)
        print(f"{sft_trainer.Colors.OKGREEN}Saved final adapter weights to {adapter_path}.{sft_trainer.Colors.ENDC}")
