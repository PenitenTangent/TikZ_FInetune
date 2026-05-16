from __future__ import annotations

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
    from .collapse_probe import run_collapse_probe
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

    def step(batch: Any, prev_grad: Any, do_update: bool) -> tuple[Any, Any, Any, Any, Any, Any]:
        if "attention_mask" in batch:
            lengths = batch["attention_mask"].sum(axis=1)
        else:
            lengths = mx.full(
                (batch["input_ids"].shape[0],),
                batch["input_ids"].shape[1],
            )

        toks = lengths.sum()
        lvalue, grad = loss_value_and_grad(model, batch)
        
        # Diagnostic: compute grad norm before clipping
        # We flatten the grad and take the norm
        flat_grad = mx.concatenate([v.reshape(-1) for _, v in mx_utils.tree_flatten(grad)])
        grad_norm = mx.linalg.norm(flat_grad)

        clip_scale = mx.array(1.0)
        clipped = mx.array(0.0)
        if args.grad_clip is not None:
            # Proper L2 norm-based clipping
            clip_scale = mx.minimum(args.grad_clip / mx.maximum(grad_norm, 1e-8), 1.0)
            clipped = mx.where(clip_scale < 0.999999, 1.0, 0.0)
            grad = mx_utils.tree_map(lambda g: g * clip_scale, grad)

        if prev_grad is not None:
            grad = mx_utils.tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = sft_trainer.average_gradients(grad)
            if grad_accum_steps > 1:
                grad = mx_utils.tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad, grad_norm, clip_scale, clipped

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0.0
    grad_accum = None
    grad_norms = 0.0
    clip_scales = 0.0
    clipped_steps = 0.0
    probe_fail_count = 0
    adapter_path = Path(args.adapter_file)
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

        lvalue, toks, grad_accum, g_norm, clip_scale, clipped = step(
            batch,
            grad_accum,
            global_it % grad_accum_steps == 0,
        )
        mx.clear_cache()
        losses += lvalue
        n_tokens += toks
        steps += 1
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
            avg_grad_norm = mx.distributed.all_sum(grad_norms, stream=mx.cpu).item() / steps
            avg_clip_scale = mx.distributed.all_sum(clip_scales, stream=mx.cpu).item() / (steps * world_size)
            clipped_step_rate = mx.distributed.all_sum(clipped_steps, stream=mx.cpu).item() / (steps * world_size)
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

            losses = 0
            n_tokens = 0
            steps = 0
            grad_norms = 0.0
            clip_scales = 0.0
            clipped_steps = 0.0
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

        # Run collapse probe every PROBE_INTERVAL steps (independent of checkpoint saves)
        # so we catch degradation well before the next save window.
        PROBE_INTERVAL = 500
        if global_it % PROBE_INTERVAL == 0 and rank == 0 and processor is not None:
            print(
                f"{sft_trainer.Colors.OKBLUE}Iter {global_it}: Running collapse probe...{sft_trainer.Colors.ENDC}",
                flush=True,
            )
            passed, failures = run_collapse_probe(model, processor, build_generation_prompt)
            if not passed:
                probe_fail_count += 1
                print(f"{sft_trainer.Colors.FAIL}WARNING: Collapse probe failed! ({probe_fail_count}/2){sft_trainer.Colors.ENDC}")
                for f in failures:
                    print(f"  - {f['prompt']}: {', '.join(f['reasons'])}")

                if probe_fail_count >= 100:
                    print(f"{sft_trainer.Colors.FAIL}FATAL: Collapse probe failed 100 times. Aborting training to save compute.{sft_trainer.Colors.ENDC}")
                    sys.exit(1)
            else:
                # High-water mark logic: decrement instead of reset
                probe_fail_count = max(0, probe_fail_count - 1)
                print(f"{sft_trainer.Colors.OKGREEN}✓ Collapse probe passed.{sft_trainer.Colors.ENDC}")

    if rank == 0:
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        sft_trainer.save_adapter(model, adapter_path)
        print(f"{sft_trainer.Colors.OKGREEN}Saved final adapter weights to {adapter_path}.{sft_trainer.Colors.ENDC}")
