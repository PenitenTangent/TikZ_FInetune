# TikZ MLX Fine-Tuning Pipeline - Ground Truth Reference

Everything in this document is verified against live code, active configs, and current manifests as of May 10, 2026.

CAUTION:
- This is a code-anchored snapshot, not a historical design memo.
- If code and this document disagree, code is authoritative.

---

## 1. Current Pipeline Topology

There are two orchestrators in the repository:

1. `tools/run_curriculum_5stage.sh` (active operational SFT path)
- Runs 5 SFT stages sequentially.
- Uses pretokenized, non-packed caches per stage.
- Uses tqdm monitor (`tools/run_with_live_progress_tqdm.py`).
- Stage chaining:
  - Resume from latest stage checkpoint when present.
  - Else resume from previous stage published adapter.
- Publishes stable stage adapters:
  - `runs/tikz_stage1_adapter.safetensors`
  - `runs/tikz_stage2_adapter.safetensors`
  - `runs/tikz_stage3_adapter.safetensors`
  - `runs/tikz_stage4_adapter.safetensors`
  - `runs/tikz_stage5_adapter.safetensors`

2. `tools/run_curriculum.sh` (legacy/alternate path)
- Runs 3 SFT stages and optional Stage-2 RL (`train-stage2`) depending on `training.stage2.enabled` or `RUN_RL`.
- Not the same flow as `run_curriculum_5stage.sh`.

The current workstream in this repo uses the 5-stage SFT pipeline.

---

## 2. Dataset and Split Ground Truth

Source dataset:
- `nllg/DaTikZ-V4` (HuggingFace)

Preparation manifest (`data/manifests/datikz_v4_prepare_manifest.json`):
- `total_seen`: 427753
- `total_written`: 302382
- `truncated_records`: 25359
- `max_context_tokens`: 1536

Current split manifest (`data/manifests/prepared_split_manifest.json`):
- `total_records`: 238009
- `train_records`: 202468
- `val_records`: 23728
- `gold_eval_records`: 11813
- `split_seed`: 17
- `source_train_path`: `data/prepared/all_prepared_sft_clean.jsonl`
- `source_stage2_path`: `data/prepared/all_prepared_stage2.jsonl`

These values supersede older counts in previous architecture notes.

---

## 3. Prompt and Record Construction

Code path:
- `src/tikz_mlx/dataset.py`
- `src/tikz_mlx/prompting.py`

Facts:
1. `sample_to_training_record()` requires non-empty description.
2. It enriches metadata with:
- `generation_mode` from content detection
- `geometry_hints`:
  - `tikz_libraries` (if found)
  - literal-coordinate `bounding_box` (if found)
3. If `\begin{document}` is present in normalized code:
- User prompt includes preamble up to `\begin{document}`.
- Assistant target contains body only, then closing fence.
4. Prompt contract comes from `build_generation_prompt()` and opens with ` ```latex `.

---

## 4. Normalization Pipeline (Current)

Code path:
- `src/tikz_mlx/normalize.py`

`normalize_tikz()` currently does:
1. Strip inline comments.
2. Strip external dependency lines (`\input`, `\include`, `\includegraphics`).
3. Ensure standalone document via `ensure_standalone_document()`:
- Uses `\documentclass{article}` (not standalone class).
- Includes preview/tikz packages and detected extras.
- Unwraps nested document artifacts.
- Heals broken pgfplots forms.
- Heals coordinate math.
- Heals missing semicolons.
- Auto-closes common environments.
- Rewrites `\tikzstyle` to `\tikzset`.
4. Strip default TikZ options.
5. Quantize long floats.
6. Remove duplicate command lines (statement-level dedup).

This is the canonical behavior for both training data normalization and evaluation normalization paths that call `normalize_tikz()`.

---

## 5. Tokenization and Packing Reality

### 5.1 Pretokenization

Code path:
- `tools/pretokenize_dataset.py`

Important current behavior:
1. Uses `transformers.AutoTokenizer` directly.
2. Applies chat template with `tokenize=False`, then explicit encode.
3. Hard-skips samples whose tokenized length exceeds `--max-tokens`.
4. Writes tokenized object array `.npy` and `<stem>_audit.json`.

### 5.2 Packing

Code path:
- `tools/pack_tokenized_dataset.py`

Produces:
- packed ids `.npy`
- masks `_masks.npy`
- boundaries `_boundaries.npy`
- optional reward/syntax sidecars
- `_audit.json`

Current 5-stage configs do not use packed caches (they use `pretokenized_cache_path`, with `pretokenized_packed_cache_path: null`).

---

## 6. Training Core Behavior

Code path:
- `src/tikz_mlx/train.py`
- `src/tikz_mlx/adapter_config_io.py`

Verified behavior:
1. Completion-mask preflight is enforced for unpacked datasets when enabled by config.
2. Loss normalization uses completion-effective masking (`completion_effective_mask_v1`).
3. Resume adapter compatibility check is strict for LoRA hyperparameters:
- rank
- alpha
- dropout
4. If `lora_num_layers` is set, trainer unwraps LoRA from early layers and keeps LoRA only on last N layers.
5. Telemetry can be written at phase boundaries (`phase_boundary_telemetry.json`).

---

## 7. Active 5-Stage SFT Configs

Active stage configs:
- `configs/curriculum_stage1.yaml`
- `configs/curriculum_stage2.yaml`
- `configs/curriculum_stage3.yaml`
- `configs/curriculum_stage4.yaml`
- `configs/curriculum_stage5.yaml`

Current stage iteration budgets:
- Stage 1: 486
- Stage 2: 361
- Stage 3: 153
- Stage 4: 120
- Stage 5: 100

Current LoRA pattern across stages:
- rank: 24
- alpha: 48
- dropout: 0.0
- layers:
  - Stage 1-4: 28
  - Stage 5: 20

Current context limits by stage:
- Stage 1: 768
- Stage 2: 1024
- Stage 3: 1536
- Stage 4: 1792
- Stage 5: 1280

Note on syntax weighting:
- Configs currently set `syntax_weighted_loss: false` in stage YAMLs.
- Trainer code does support syntax weighting with pretokenized cache path when enabled.

---

## 8. Operational Resume and Adapter Shim

Code path:
- `tools/run_stage.sh`
- `src/tikz_mlx/adapter_config_io.py`

`run_stage.sh` writes `runs/adapter_config.json` from current stage LoRA settings before training.
This supports resume flows that provide safetensors paths where mlx-vlm expects directory-side config context.

Resume precedence in stage runner:
1. Latest checkpoint in stage dir.
2. Previous stage published adapter.
3. Fresh start.

---

## 9. Evaluation Pipeline (A/B)

Code path:
- `tools/ab_eval.py`
- post-train hook in `src/tikz_mlx/cli.py`

Current facts:
1. `ab_eval.py` default `--num-samples` is 120.
2. Post-training hook (`tikz_mlx.cli train`) runs A/B automatically unless `--skip-post-ab-eval`.
3. Post-train hook currently uses:
- `--num-samples 50`
- `--checkpoint-dir <adapter parent>`
- `--max-tokens 2048`
4. `ab_eval.py` reports both per-sample outputs and summary metrics including:
- compile rate
- substantive rate
- repetition-loop rate
- code length

---

## 10. About `configs/lora_prod.yaml`

`configs/lora_prod.yaml` currently contains an ablation-style baseline profile, including:
- model id: `mlx-community/gemma-4-e4b-it-6bit`
- lora rank/alpha/dropout: 8/16/0.1
- lora layers: 16
- packed cache enabled

This does not represent the active 5-stage curriculum settings.
For current staged training, use the `configs/curriculum_stage*.yaml` files.

---

## 11. Rebuild Checklist (Current 5-Stage Path)

1. Prepare/split data
- `make prep-dataset`
- `make split-dataset`

2. Build curriculum splits
- `python tools/build_curriculum.py ...` (or use existing prepared curriculum JSONLs)

3. Pretokenize each stage
- `python tools/pretokenize_dataset.py --model-id mlx-community/gemma-4-e4b-it-6bit --dataset data/prepared/curriculum/train_stageN.jsonl --output data/prepared/curriculum/train_train_stageN_tokenized.npy --max-tokens <stage_limit>`

4. Run stage pipeline
- Single stage: `bash tools/run_stage.sh <1-5> [--resume]`
- Full 5-stage: `bash tools/run_curriculum_5stage.sh`

5. Evaluate adapters
- `python tools/ab_eval.py --config configs/curriculum_stage5.yaml --adapter-path runs/tikz_stage5_adapter.safetensors --num-samples <N> --seed <S> --max-tokens 2048`

---

## 12. Known Current Risks

1. Legacy and active orchestrators coexist (`run_curriculum.sh` vs `run_curriculum_5stage.sh`), and they represent different training flows.
2. `lora_prod.yaml` and curriculum stage configs can diverge significantly; avoid mixing assumptions between them.

---

## 13. Verification Scope

This reference reflects the following live artifacts as ground truth:
- manifests in `data/manifests/`
- stage configs in `configs/curriculum_stage*.yaml`
- orchestration scripts in `tools/run_stage.sh` and `tools/run_curriculum_5stage.sh`
- trainer and adapter validation code in `src/tikz_mlx/train.py` and `src/tikz_mlx/adapter_config_io.py`
- evaluation code in `tools/ab_eval.py` and `src/tikz_mlx/cli.py`

End of document.
