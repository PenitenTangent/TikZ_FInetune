from __future__ import annotations

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
    print(f"{sft_trainer.Colors.HEADER}Starting training..., iterations: {args.iters}{sft_trainer.Colors.ENDC}")

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

    loss_fn_partial = partial(
        loss_fn,
        train_on_completions=train_on_completions,
        assistant_id=assistant_id,
    )
    loss_value_and_grad = nn.value_and_grad(model, loss_fn_partial)
    state = [model.state, optimizer.state, mx.random.state]
    eval_milestones = frozenset(int(x) for x in getattr(args, "_tikz_eval_at", frozenset()))

    def step(batch: Any, prev_grad: Any, do_update: bool) -> tuple[Any, Any, Any, Any]:
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
        flat_grad = mx.concatenate([g.reshape(-1) for g in sft_trainer.tree_flatten(grad)])
        grad_norm = mx.linalg.norm(flat_grad)

        if args.grad_clip is not None:
            grad = sft_trainer.tree_map(lambda g: mx.clip(g, -args.grad_clip, args.grad_clip), grad)

        if prev_grad is not None:
            grad = sft_trainer.tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = sft_trainer.average_gradients(grad)
            if grad_accum_steps > 1:
                grad = sft_trainer.tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad, grad_norm

    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0.0
    grad_accum = None
    grad_norms = 0.0
    probe_fail_count = 0

    for it, batch in zip(
        range(1, args.iters + 1),
        sft_trainer.iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        tic = time.perf_counter()

        if val_dataset is not None and it in eval_milestones:
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
                    f"{sft_trainer.Colors.OKCYAN}Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s{sft_trainer.Colors.ENDC}",
                    flush=True,
                )
            tic = time.perf_counter()

        lvalue, toks, grad_accum, g_norm = step(batch, grad_accum, it % grad_accum_steps == 0)
        mx.clear_cache()
        losses += lvalue
        n_tokens += toks
        steps += 1
        grad_norms += g_norm
        mx.eval(state, losses, n_tokens, grad_accum, grad_norms)
        train_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            avg_grad_norm = mx.distributed.all_sum(grad_norms, stream=mx.cpu).item() / steps
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
                    f"Iter {it}: Train loss {sft_trainer.Colors.OKGREEN}{train_loss:.8f}{sft_trainer.Colors.ENDC}, "
                    f"Grad Norm {avg_grad_norm:.4f}, "
                    f"LR {learning_rate:.3e}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            losses = 0
            n_tokens = 0
            steps = 0
            grad_norms = 0.0
            train_time = 0.0

        if it % args.steps_per_save == 0 and rank == 0:
            sft_trainer.save_adapter(model, args.adapter_file)
            checkpoint = Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            sft_trainer.save_adapter(model, checkpoint)
            print(
                f"{sft_trainer.Colors.OKBLUE}Iter {it}: Saved adapter to {checkpoint}. Running collapse probe...{sft_trainer.Colors.ENDC}",
                flush=True,
            )
            
            # Upgrade 5: Online generation probe
            if processor is not None:
                passed, failures = run_collapse_probe(model, processor, build_generation_prompt)
                if not passed:
                    probe_fail_count += 1
                    print(f"{sft_trainer.Colors.FAIL}WARNING: Collapse probe failed! ({probe_fail_count}/2){sft_trainer.Colors.ENDC}")
                    for f in failures:
                        print(f"  - {f['prompt']}: {', '.join(f['reasons'])}")
                    
                    if probe_fail_count >= 2:
                        print(f"{sft_trainer.Colors.FAIL}FATAL: Collapse probe failed twice. Aborting training to save compute.{sft_trainer.Colors.ENDC}")
                        sys.exit(1)
                else:
                    probe_fail_count = 0
                    print(f"{sft_trainer.Colors.OKGREEN}✓ Collapse probe passed.{sft_trainer.Colors.ENDC}")

    if rank == 0:
        sft_trainer.save_adapter(model, args.adapter_file)
        print(f"{sft_trainer.Colors.OKGREEN}Saved final adapter weights to {args.adapter_file}.{sft_trainer.Colors.ENDC}")

    if rank == 0:
        sft_trainer.save_adapter(model, args.adapter_file)
        print(f"{sft_trainer.Colors.OKGREEN}Saved final adapter weights to {args.adapter_file}.{sft_trainer.Colors.ENDC}")
