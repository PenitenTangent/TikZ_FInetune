#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import yaml
import numpy as np
import sys


def allocate_stage_iters(stage_sizes: list[int], total_iters: int, n_records: int) -> list[int]:
    """Split total_iters across stages proportionally; every non-empty stage gets at least 1 when possible."""
    k = len(stage_sizes)
    out = [0] * k
    if total_iters <= 0 or n_records <= 0:
        return out
    non_empty = [i for i in range(k) if stage_sizes[i] > 0]
    if not non_empty:
        return out
    if total_iters < len(non_empty):
        for j in range(total_iters):
            out[non_empty[j]] = 1
        return out
    weights = [stage_sizes[i] / n_records for i in range(k)]
    remaining = total_iters - len(non_empty)
    for i in non_empty:
        out[i] = 1
    extra = [remaining * weights[i] for i in non_empty]
    floors = [int(x) for x in extra]
    for idx, i in enumerate(non_empty):
        out[i] += floors[idx]
    rem = remaining - sum(floors)
    order = sorted(range(len(non_empty)), key=lambda t: extra[t] - floors[t], reverse=True)
    for j in range(rem):
        out[non_empty[order[j % len(order)]]] += 1
    return out


def require_stage3_pretokenize_long_coverage(*, audit_path: pathlib.Path, min_fraction_ge_1024: float) -> None:
    """Fail fast when stage-3 kept rows rarely reach long context (metadata before packing)."""
    if min_fraction_ge_1024 <= 0:
        return
    if not audit_path.exists():
        raise RuntimeError(f"Missing pretokenize audit after tokenization: {audit_path}")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    ge_frac = float(audit_payload.get("kept_length_ge_fractions", {}).get("1024", 0.0))
    if ge_frac < min_fraction_ge_1024:
        raise RuntimeError(
            "Stage 3 pretokenize long-context coverage is below the configured floor — "
            f"fraction_kept_ge_1024={ge_frac:.4f}, required={min_fraction_ge_1024:.4f}. "
            "The phase may not train meaningfully on long TikZ. "
            "Inspect the stage-3 JSONL slice and metadata token lengths, "
            "or lower --stage3-min-kept-fraction-ge-1024 (or pass --disable-stage3-long-coverage-check) for debugging."
        )


def filter_val_jsonl_by_min_metadata_tokens(
    *,
    val_src: pathlib.Path,
    val_out: pathlib.Path,
    min_metadata_tokens: int,
) -> tuple[int, int]:
    """Write validation rows whose metadata.token_length >= min. Returns (kept, scanned)."""
    kept = 0
    scanned = 0
    val_out.parent.mkdir(parents=True, exist_ok=True)
    with val_src.open("r", encoding="utf-8") as src, val_out.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            raw = meta.get("token_length")
            try:
                tl = int(raw)
            except (TypeError, ValueError):
                continue
            if tl >= min_metadata_tokens:
                dst.write(json.dumps(record) + "\n")
                kept += 1
    return kept, scanned


def k_means_1d(data, k, iterations=100):
    """Simple 1D K-Means implementation for partitioning token lengths into stages.

    Returns the sorted list of centroids.
    """
    if not data: return []
    data = np.array(data)
    centroids = np.linspace(np.min(data), np.max(data), k)
    for _ in range(iterations):
        distances = np.abs(data[:, None] - centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean() if len(data[labels == i]) > 0 else centroids[i] for i in range(k)])
        if np.all(centroids == new_centroids): break
        centroids = new_centroids
    return sorted(centroids)


def load_train_records(dataset_path: pathlib.Path, max_examples: int | None = None) -> tuple[list, list]:
    """Read JSONL training rows in file order. If max_examples is set, stop after that many non-empty records."""
    records: list = []
    token_lengths: list = []
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            token_lengths.append(rec.get("metadata", {}).get("token_length", 512))
            if max_examples is not None and len(records) >= max_examples:
                break
    return records, token_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--val-dataset", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data/prepared/curriculum")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable for pretokenize/pack subprocesses (default: sys.executable).",
    )
    parser.add_argument(
        "--stage3-min-kept-fraction-ge-1024",
        type=float,
        default=0.02,
        help="For stage 3 only: minimum fraction of pretokenized (kept) rows with length>=1024; "
        "fails fast if the long-context slice is too small. Use 0 to disable.",
    )
    parser.add_argument(
        "--disable-stage3-long-coverage-check",
        action="store_true",
        help="Skip stage-3 pretokenize length distribution check (not recommended for production).",
    )
    parser.add_argument(
        "--phase-aware-val-subset",
        action="store_true",
        help="Per stage, write a validation JSONL subset with metadata.token_length >= 0.45 * max_context.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        metavar="N",
        help="Load at most N training rows from --dataset (first non-empty lines, file order).",
    )
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be >= 1 when set")
    py_exe = args.python or sys.executable

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)

    assistant_id = base_config.get("training", {}).get("assistant_id", 4368)

    print(f"Loading {args.dataset}...")
    records, token_lengths = load_train_records(
        pathlib.Path(args.dataset),
        max_examples=args.max_examples,
    )
    cap_note = f" (capped by --max-examples={args.max_examples})" if args.max_examples is not None else ""
    print(f"Loaded {len(records)} examples{cap_note}. Calculating K-Means boundaries for 3 stages...")
    centroids = k_means_1d(token_lengths, 3)
    boundaries = []
    if len(centroids) >= 2:
        boundaries = [(centroids[i] + centroids[i+1])/2 for i in range(len(centroids)-1)]
    
    stages = [[], [], []]
    for rec, length in zip(records, token_lengths):
        if not boundaries or length <= boundaries[0]:
            stages[0].append(rec)
        elif length <= boundaries[1]:
            stages[1].append(rec)
        else:
            stages[2].append(rec)

    stage_sizes = [len(s) for s in stages]
    stage_iters_list = allocate_stage_iters(stage_sizes, args.iters, len(records))

    for i, stage_records in enumerate(stages):
        stage_num = i + 1
        lengths = [r.get("metadata", {}).get("token_length", 512) for r in stage_records]
        if not lengths: continue
        
        p95 = int(np.percentile(lengths, 95))
        max_context = ((p95 + 127) // 128) * 128
        max_context = max(512, min(max_context, 1536))
        stage_iters = stage_iters_list[i]
        
        if max_context <= 768:
            grad_accum, val_batches, lr = 4, 25, 3e-5
        elif max_context <= 1024:
            grad_accum, val_batches, lr = 2, 10, 2e-5
        else:
            grad_accum, val_batches, lr = 1, 5, 1e-5

        print(f"\n--- Stage {stage_num} ---")
        print(f"Examples: {len(stage_records)} ({len(stage_records)/len(records)*100:.1f}%)")
        print(f"Dynamic max_context_tokens: {max_context}")
        print(f"Iteration Budget: {stage_iters}")
        print(f"Hardware Profile -> Accum: {grad_accum}, Val Batches: {val_batches}")

        stage_ds_path = out_dir / f"train_stage{stage_num}.jsonl"
        packed_cache_path = out_dir / f"train_stage{stage_num}_packed.npy"
        config_path = pathlib.Path("configs") / f"curriculum_stage{stage_num}.yaml"
        
        if not args.dry_run:
            with open(stage_ds_path, "w", encoding="utf-8") as f:
                for example in stage_records:
                    f.write(json.dumps(example) + "\n")
            
            tokenized_tmp = out_dir / f"train_stage{stage_num}_tokenized.npy"
            
            # Step 1: Tokenize
            print(f"Running tokenization for Stage {stage_num}...")
            subprocess.run([
                py_exe, "tools/pretokenize_dataset.py",
                "--model-id", base_config["model"]["model_id"],
                "--dataset", str(stage_ds_path),
                "--output", str(tokenized_tmp),
                "--max-tokens", str(max_context)
            ], check=True)

            audit_path = tokenized_tmp.with_name(tokenized_tmp.stem + "_audit.json")
            if stage_num == 3 and not args.disable_stage3_long_coverage_check:
                require_stage3_pretokenize_long_coverage(
                    audit_path=audit_path,
                    min_fraction_ge_1024=float(args.stage3_min_kept_fraction_ge_1024),
                )

            # Step 2: Pack
            print(f"Running sequence packing for Stage {stage_num}...")
            subprocess.run([
                py_exe, "tools/pack_tokenized_dataset.py",
                "--input", str(tokenized_tmp),
                "--output", str(packed_cache_path),
                "--max-tokens", str(max_context),
                "--assistant-token", str(assistant_id)
            ], check=True)
            
            if tokenized_tmp.exists(): tokenized_tmp.unlink()
                
            stage_config = json.loads(json.dumps(base_config))
            stage_config["model"]["max_context_tokens"] = max_context
            stage_config["memory"]["gradient_accumulation_steps"] = grad_accum
            stage_config["training"]["lora_num_layers"] = 28
            stage_config["training"]["iters"] = stage_iters
            stage_config["training"]["val_batches"] = val_batches
            stage_config["training"]["learning_rate"] = lr
            stage_config["training"]["train_dataset_path"] = str(stage_ds_path.resolve())
            val_path_write = pathlib.Path(args.val_dataset).resolve()
            if args.phase_aware_val_subset and val_path_write.suffix.lower() == ".jsonl":
                min_meta = max(1, int(max_context * 0.45))
                val_slice = out_dir / f"val_stage{stage_num}_min_meta_{min_meta}.jsonl"
                kept_val, scanned_val = filter_val_jsonl_by_min_metadata_tokens(
                    val_src=val_path_write,
                    val_out=val_slice,
                    min_metadata_tokens=min_meta,
                )
                if kept_val <= 0:
                    print(
                        f"[warn] Phase-aware val subset empty for stage {stage_num} "
                        f"(min_metadata={min_meta}, scanned={scanned_val}); using full validation file."
                    )
                else:
                    print(
                        f"Phase-aware validation subset: stage {stage_num}, kept={kept_val}/{scanned_val}, "
                        f"path={val_slice}"
                    )
                    val_path_write = val_slice.resolve()

            stage_config["training"]["val_dataset_path"] = str(val_path_write)
            stage_config["training"]["pretokenized_packed_cache_path"] = str(packed_cache_path.resolve())
                
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(stage_config, f, sort_keys=False)

if __name__ == "__main__":
    main()
