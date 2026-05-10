#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import sys
import numpy as np
from tqdm import tqdm
from tikz_mlx.prompting import prompt_template_sha256, PROMPT_CONTRACT_VERSION


def _file_sha256(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _ids_sha256(values: list[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize a JSONL dataset for TikZ finetuning.")
    parser.add_argument("--model-id", type=str, required=False, help="Model ID to load processor from.")
    parser.add_argument("--config", type=str, required=False,
                        help="Path to a curriculum stage YAML config. When provided, model_id and "
                             "max_context_tokens are read from the config and override --model-id / --max-tokens.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to save tokenized data (.npy).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens for truncation. Defaults to model.max_context_tokens from --config, or 2048.")
    parser.add_argument("--prompt-contract-version", default="tikz_partial_decode_v1")
    parser.add_argument("--normalization-config-hash", default="")
    parser.add_argument("--disabled-rules", default="", help="Comma-separated disabled normalization/filter rules.")
    
    args = parser.parse_args()

    # Load config-driven overrides
    config_model_id: str | None = None
    config_max_tokens: int | None = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        config_model_id = cfg.get("model", {}).get("model_id")
        config_max_tokens = int(cfg.get("model", {}).get("max_context_tokens", 2048))
        print(f"[config] Loaded from {args.config}: model_id={config_model_id}, max_context_tokens={config_max_tokens}")

    model_id: str = args.model_id or config_model_id or ""
    if not model_id:
        print("Error: provide --model-id or --config with a model.model_id.", file=sys.stderr)
        sys.exit(1)

    # max-tokens precedence: explicit CLI > config > default 2048
    if args.max_tokens is not None:
        effective_max_tokens = args.max_tokens
    elif config_max_tokens is not None:
        effective_max_tokens = config_max_tokens
    else:
        effective_max_tokens = 2048
    print(f"[pretokenize] Effective max_tokens: {effective_max_tokens}")
    
    # Use AutoTokenizer directly — mlx_vlm.load_processor always attempts to load
    # a video_preprocessor_config.json that doesn't exist for text-only models,
    # producing a spurious WARNING on every run.
    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    dataset_path = pathlib.Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset {args.dataset} not found.")
        sys.exit(1)

    print(f"Tokenizing {args.dataset}...")
    tokenized_samples = []
    kept_sample_ids: list[str] = []
    skipped_sample_ids: list[str] = []
    token_lengths: list[int] = []
    
    with dataset_path.open("r", encoding="utf-8") as f:
        # Count lines for tqdm
        total_lines = sum(1 for _ in f)
        f.seek(0)
        
        for row_index, line in enumerate(tqdm(f, total=total_lines)):
            record = json.loads(line)
            sample_id = str(record.get("sample_id", f"row_{row_index:06d}"))
            messages = record.get("messages")
            if not messages:
                skipped_sample_ids.append(sample_id)
                continue
            
            # Step 1: Prompt Contamination Preflight
            # We look for actual usage (usually followed by { or [) rather than just mentions.
            toxic_assistant_tokens = [
                "\\documentclass", "\\usepackage", "\\PreviewEnvironment",
                "\\begin{document}", "\\end{document}"
            ]
            toxic_user_tokens = [
                "\\PreviewEnvironment{", "\\usepackage[active,tightpage]{preview}"
            ]
            
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                
                if msg["role"] == "assistant":
                    for token in toxic_assistant_tokens:
                        if token in content:
                            print(f"\nERROR: Toxic token '{token}' found in assistant target for sample {sample_id}")
                            print("Aborting. Data contract violation (prompt contamination).")
                            sys.exit(1)
                elif msg["role"] == "user":
                    for token in toxic_user_tokens:
                        if token in content:
                            print(f"\nERROR: Toxic token '{token}' found in user prompt for sample {sample_id}")
                            print("Aborting. Data contract violation (prompt contamination).")
                            sys.exit(1)
            
            # Flatten structured content
            flat_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                flat_messages.append({"role": msg["role"], "content": content})

            # Step 1: Format to a plain string using the chat template.
            # tokenize=False guarantees a str return, no BatchEncoding ambiguity.
            text = tokenizer.apply_chat_template(
                flat_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Step 2: Encode with truncation explicitly OFF to see the true length.
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,  # chat template already includes BOS/EOS
                truncation=False,
            )

            if effective_max_tokens and len(token_ids) > effective_max_tokens:
                skipped_sample_ids.append(sample_id)
                continue

            tokenized_samples.append(np.array(token_ids, dtype=np.int32))
            kept_sample_ids.append(sample_id)
            token_lengths.append(len(token_ids))

    skipped = total_lines - len(tokenized_samples)
    print(f"\n--- Tokenization Report ---")
    print(f"Total processed: {total_lines}")
    print(f"Clean samples saved: {len(tokenized_samples)}")
    print(f"Discarded (too long): {skipped}")
    print(f"---------------------------\n")
    # Using object array to store varying length sequences efficiently
    data = np.array(tokenized_samples, dtype=object)
    output_path = pathlib.Path(args.output)
    np.save(output_path, data)
    n_kept = len(token_lengths)
    length_ge_fractions: dict[str, float] = {}
    if n_kept > 0:
        for thr in (512, 768, 1024, 1280, 1536):
            count_ge = sum(1 for length in token_lengths if length >= thr)
            length_ge_fractions[str(thr)] = count_ge / n_kept

    audit = {
        "source_jsonl_path": str(dataset_path.expanduser().resolve()),
        "source_jsonl_sha256": _file_sha256(dataset_path),
        "source_row_count": total_lines,
        "tokenized_row_count": len(tokenized_samples),
        "skipped_row_count": len(skipped_sample_ids),
        "kept_sample_ids_sha256": _ids_sha256(kept_sample_ids),
        "skipped_sample_ids": skipped_sample_ids,
        "model_id": model_id,
        "tokenizer_id": model_id,
        "max_tokens": effective_max_tokens,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "prompt_template_sha256": prompt_template_sha256(),
        "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest() if hasattr(tokenizer, "chat_template") else None,
        "assistant_boundary_rule": "assistant role completion only after opened latex fence",
        "normalization_config_hash": args.normalization_config_hash,
        "disabled_rules": [value for value in args.disabled_rules.split(",") if value],
        "token_length_histogram": {
            "min": min(token_lengths) if token_lengths else 0,
            "max": max(token_lengths) if token_lengths else 0,
            "mean": float(np.mean(token_lengths)) if token_lengths else 0.0,
            "p95": float(np.percentile(token_lengths, 95)) if token_lengths else 0.0,
            "p99": float(np.percentile(token_lengths, 99)) if token_lengths else 0.0,
        },
        "kept_length_ge_fractions": length_ge_fractions,
    }
    audit_path = output_path.with_name(output_path.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Audit written to {audit_path}")
    print("Done.")

if __name__ == "__main__":
    main()
