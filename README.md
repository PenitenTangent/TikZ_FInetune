# TikZ MLX Pipeline

Apple-Silicon-native text-to-TikZ pipeline for a 24 GB M4 Pro Mac mini.

This repository is structured to build the full system around fine-tuning first, while intentionally deferring the final long-running fine-tuning job until the rest of the pipeline is stable. The current implementation focus is:

- streaming data preparation for DaTikZ-style corpora
- safe compilation and structured compiler feedback with `tectonic`
- MLX / `mlx-vlm` inference adapters for Gemma 4 style multimodal models
- split repair loops for syntax fixes and visual fixes
- evaluation and reward interfaces
- a guarded training dry-run harness instead of a long finetuning run

## Design principles

- No CUDA, Triton, or PyTorch runtime path in project code
- One large model resident at a time on the 24 GB machine
- Immutable canonical TikZ output; debug overlays are derived artifacts
- Visual repair only after a successful render exists
- Training code is built now, but the final long run stays opt-in

## Layout

```text
configs/            Runtime and training configuration
src/tikz_mlx/       Package code
tests/              Focused regression tests for core logic
```

## Quick start

```bash
make install
make prep
make check-dataset
make prep-dataset
make split-dataset
make train
```

`make train` is intentionally wired to the training dry-run path until you explicitly allow a full run in config or CLI flags.

## Dataset preparation

Run a fast readiness probe before full streaming preparation:

`tikz-mlx check-dataset --dataset-id nllg/DaTikZ-V4 --split train --sample-limit 128`

Then prepare data with:

Use `tikz-mlx prepare-dataset --config configs/lora_prod.yaml --overwrite` to stream `nllg/DaTikZ-V4` from Hugging Face into:

- `data/prepared/train.jsonl` for SFT
- `data/prepared/train_stage2.jsonl` for stage 2
- `data/prepared/images/` for reference images
- `data/manifests/datikz_v4_prepare_manifest.json` for preparation stats

The pipeline rejects malformed rows defensively (`missing_tikz_code`, `missing_description`) and only attaches image paths when image writes succeed.
Preparation now also records per-sample `token_length` / `is_truncated` metadata and writes truncation stats to `data/manifests/datikz_v4_prepare_manifest.json` so the effective context limit is visible before training.

For production reward-weighted SFT, score the prepared dataset before tokenization and packing. The default scoring backend renders each compiling ground-truth figure and compares it to the prompt description with a SentenceTransformers CLIP model:

`python tools/score_training_dataset.py --config configs/lora_prod.yaml --input data/prepared/train.jsonl --output data/prepared/train_scored.jsonl --workers 8 --alignment-backend sentence-transformers-clip`

Install the optional alignment dependency before using that backend:

`pip install -e ".[alignment]"`

Use `--alignment-backend none` only for compile-success-only weights. Truncated records are skipped and receive `sample_weight: 0.0`.

Any change to normalization, `sample_to_training_record()`, mode tags, geometry hints, prompt structure, tokenizer/model id, context length, or scoring metadata invalidates downstream tokenized and packed caches. Re-run scoring, pretokenization, and packing instead of reusing stale `*_tokenized.npy`, `*_packed.npy`, masks, or sidecar weight arrays.

To append a local figure, run `tikz-mlx add-figure --config configs/lora_prod.yaml --tex-file path/to/figure.tex --image-file path/to/figure.png --description "..."`.

After preparation, create leakage-safe train/validation/gold splits (grouped by normalized content hash) with:

`tikz-mlx split-dataset --config configs/lora_prod.yaml --val-fraction 0.10 --gold-eval-fraction 0.05 --overwrite`

This writes SFT and stage-2 splits plus `data/manifests/prepared_split_manifest.json`.

## Inference profile

The production config now uses bounded best-of-N generation (`initial_candidates=4`) with compile-aware selection and phase-specific decoding for initial generation, compile repair, and visual repair. This improves compile reliability while keeping retry budgets and Apple-Silicon memory limits conservative.

## Stage 2 telemetry and gating

Stage-2 training now emits run-scoped metrics JSONL telemetry (`runs/stage2_checkpoints/metrics_<run_id>.jsonl` by default) and applies metric gates before promoting interval checkpoints. The promotion thresholds are controlled by:

- `training.stage2.promotion_min_reward`
- `training.stage2.promotion_min_compile_rate`

Reward shaping floors (`reward_format_floor`, `reward_compile_floor`) are also configurable to reduce sparse all-zero rewards early in training. A dead-signal watchdog can abort early when both format rejects and truncation stay saturated across the first save windows:

- `training.stage2.dead_signal_watchdog_enabled`
- `training.stage2.dead_signal_watchdog_windows`
- `training.stage2.dead_signal_watchdog_min_format_reject_rate`
- `training.stage2.dead_signal_watchdog_min_truncated_rate`

## Full finetune run command

For a full stage1 + stage2 run over the prepared train and validation datasets with persistent logs, use:

`make full-finetune`

The runner script will:

- create a run-specific log directory under `runs/logs/`
- write `stage1.log` and `stage2.log` for detailed later analysis
- save a run-local full-training config copy (`config_full.yaml`)
- create a stage2-compatible resume adapter directory from stage1 output
- copy stage2 telemetry JSONL into the run log directory when available

Progress bars are shown during training:

- Stage1 progress from the `mlx-vlm` trainer
- Stage2 progress via the `Stage2 training` tqdm bar

## Phase1-only strict launch

To run Stage1 only (no Stage2), enforce strict base-vs-stage1 quality gating, and promote only on gate pass:

`make phase1-finetune`

This launcher (`tools/run_phase1_finetune.sh`) will:

- force `training.allow_full_training=true`
- force `training.stage2.enabled=false`
- run Stage1 finetuning
- run strict Stage1-vs-base A/B evaluation on `gold_eval.jsonl`
- run the promotion gate using strict substantive metrics
- promote the Stage1 checkpoint to `runs/sft_final.safetensors` and `runs/policy_init.safetensors` only if the gate passes

Default strict gate thresholds are environment-overridable:

- `MIN_SUBSTANTIVE_COMPILE_DELTA`
- `MIN_SUBSTANTIVE_TIKZ_DELTA`
- `MIN_STAGE1_SUBSTANTIVE_COMPILE_RATE`
- `MIN_STAGE1_SUBSTANTIVE_TIKZ_RATE`
- `AB_HYBRID_VISUAL_THRESHOLD`

The strict A/B evaluation now scores promotion on Phase B visual similarity only. Tune `AB_HYBRID_VISUAL_THRESHOLD` from baseline reports before a production promotion run.

## Packed dataset audit

`tools/pack_tokenized_dataset.py` now emits `*_audit.json` alongside the packed ids, masks, and boundaries files. Training refuses to start against a packed cache if:

- the audit file is missing
- the audit hash does not match the live packed artifacts
- the assistant token does not match config
- the marker hit-rate or masked-token fraction fall below the configured floors

This moves mask integrity failure to pack time instead of after model load.

The packer can also emit:

- `*_reward_weights.npy` from `metadata.sample_weight`
- `*_syntax_weights.npy` when `--emit-syntax-weights --model-id ...` is used

These files are optional. For the default production config, syntax weighting can run from an in-memory token lookup without writing a sidecar array.

## Weighted SFT loss

Stage-1 training now supports:

- reward-weighted CE via `training.reward_weighted_loss`
- syntax-aware token weighting via `training.syntax_weighted_loss`

The default production config leaves reward weighting off only until a scored dataset is prepared, and enables syntax weighting because it does not require a second dataset pass.

## Render determinism

Evaluation and reward scoring now load pinned rasterization settings from `configs/render_config.yaml`. This keeps the Phase B visual metric on a fixed rendering path instead of relying on machine-local defaults.
# TikZ_FInetune
