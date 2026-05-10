#!/usr/bin/env python3
"""
pack_tokenized_dataset.py

Packs a pre-tokenized .npy dataset (produced by pretokenize_dataset.py) into
fixed-length buffers for sequence packing during training.

Each packed buffer contains multiple examples concatenated together up to
--max-tokens. A companion boundaries file records the END position of each
assistant-turn START within each pack.

Outputs:
  <output_stem>.npy             - packed input_ids, shape (N_packs, max_tokens)
  <output_stem>_masks.npy       - completion mask, shape (N_packs, max_tokens)
  <output_stem>_boundaries.npy  - assistant marker positions, shape (N_packs, MAX_SEQS_PER_PACK)
                                  padded with -1 for unused slots
  <output_stem>_audit.json      - pack integrity report consumed by training startup

Usage:
  python tools/pack_tokenized_dataset.py \\
    --input data/prepared/train_tokenized.npy \\
    --output data/prepared/train_packed.npy \\
    --max-tokens 1536 \\
    --assistant-token 4368
"""
import argparse
import hashlib
import json
import pathlib
import re
import numpy as np
from tqdm import tqdm

# Maximum number of packed sequences per buffer. Packs larger than this are
# silently capped (this is a safety guard; typical packs have 3-5 sequences).
MAX_SEQS_PER_PACK = 10

# Separator token placed between packed examples so the model can learn turn
# boundaries. Using EOS (1 for Gemma) is conventional.
SEPARATOR_TOKEN = 1  # Gemma <eos>
STRUCTURAL_TOKEN_PATTERN = re.compile(r"^[{}\[\];,\\]+$")
COMMAND_TOKEN_PATTERN = re.compile(r"^\\[A-Za-z@]+$")
COORDINATE_TOKEN_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")


def _find_assistant_boundary(tokens: np.ndarray, assistant_token: int) -> int:
    """Return the index of the first occurrence of assistant_token, or -1."""
    positions = np.where(tokens == assistant_token)[0]
    return int(positions[0]) if positions.size > 0 else -1


def _file_sha256(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _pack_dataset_sha256(*paths: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def _load_record_metadata(metadata_path: pathlib.Path) -> list[dict]:
    records: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            record = dict(payload.get("metadata", {}))
            if "sample_id" in payload and "sample_id" not in record:
                record["sample_id"] = payload["sample_id"]
            records.append(record)
    return records


def _metadata_sample_id(metadata: dict) -> str | None:
    sample_id = metadata.get("sample_id")
    if sample_id is None:
        return None
    return str(sample_id)


def _align_sample_metadata(
    tokenized: np.ndarray,
    sample_metadata: list[dict],
    pretokenize_audit: dict | None,
) -> list[dict]:
    if len(sample_metadata) == len(tokenized):
        return sample_metadata

    audit_skipped_ids: set[str] = set()
    if isinstance(pretokenize_audit, dict):
        audit_tokenized_rows = pretokenize_audit.get("tokenized_row_count")
        if audit_tokenized_rows is not None and int(audit_tokenized_rows) != len(tokenized):
            raise RuntimeError(
                "Pretokenize audit does not match the tokenized array: "
                f"audit tokenized_row_count={audit_tokenized_rows} tokenized={len(tokenized)}"
            )
        audit_skipped_ids = {
            str(sample_id)
            for sample_id in pretokenize_audit.get("skipped_sample_ids", [])
            if sample_id is not None
        }

    if audit_skipped_ids:
        filtered_metadata = [
            metadata
            for metadata in sample_metadata
            if _metadata_sample_id(metadata) not in audit_skipped_ids
        ]
        if len(filtered_metadata) == len(tokenized):
            return filtered_metadata

    raise RuntimeError(
        "Metadata row count mismatch: "
        f"tokenized={len(tokenized)} metadata={len(sample_metadata)}. "
        "If the tokenized array was produced from the same JSONL but skipped long rows, "
        "pass the matching pretokenize audit so the packer can filter metadata by skipped sample_id."
    )


def _build_syntax_lookup(
    model_id: str,
    *,
    structural_weight: float,
    command_weight: float,
    coordinate_weight: float,
) -> np.ndarray:
    # Load the tokenizer directly via transformers to avoid mlx-vlm's
    # load_processor, which tries to load a video_preprocessor_config.json
    # that Gemma 4 (and most VLMs used for TikZ) does not have.
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer for '{model_id}' to build syntax weights. "
            f"Make sure the model is cached locally. Original error: {exc}"
        ) from exc

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


def pack_dataset(
    tokenized: np.ndarray,
    max_tokens: int,
    assistant_token: int,
    sample_metadata: list[dict] | None = None,
    pretokenize_audit: dict | None = None,
    syntax_lookup: np.ndarray | None = None,
    pad_token: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, int | float],
]:
    """
    Greedily pack tokenized sequences into fixed-length buffers.

    Returns:
        packed_ids:    int32 array (N_packs, max_tokens)
        packed_masks:   uint8 array (N_packs, max_tokens) - 1 for completion, 0 for prompt/pad
        boundaries:    int32 array (N_packs, MAX_SEQS_PER_PACK)
    """
    packed_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    boundary_rows: list[list[int]] = []
    reward_weight_rows: list[np.ndarray] = []
    syntax_weight_rows: list[np.ndarray] = []

    current_buf: list[int] = []
    current_mask: list[int] = []
    current_bounds: list[int] = []
    current_reward_weight: list[float] = []
    current_syntax_weight: list[float] = []
    input_sequences = 0
    marker_hit_sequences = 0
    truncated_sequences = 0

    if sample_metadata is not None:
        sample_metadata = _align_sample_metadata(tokenized, sample_metadata, pretokenize_audit)

    for row_index, raw_tokens in enumerate(tqdm(tokenized, desc="Packing")):
        input_sequences += 1
        tokens = raw_tokens.astype(np.int32).tolist()
        metadata = sample_metadata[row_index] if sample_metadata is not None else {}

        # Find assistant start
        asst_positions = [i for i, t in enumerate(tokens) if t == assistant_token]
        boundary = asst_positions[0] if asst_positions else -1
        if boundary >= 0:
            marker_hit_sequences += 1

        # Create mask for THIS example (0 for prompt up to boundary, 1 after)
        # Note: boundary is the index of the marker token itself.
        # We start unmasking at boundary + 1.
        local_mask = [0] * len(tokens)
        if boundary >= 0:
            for i in range(boundary + 1, len(tokens)):
                local_mask[i] = 1

        local_reward_weight = None
        if sample_metadata is not None:
            sample_weight = float(metadata.get("sample_weight", 1.0))
            if bool(metadata.get("is_truncated", False)):
                sample_weight = 0.0
            local_reward_weight = [0.0] * len(tokens)
            if boundary >= 0:
                for i in range(boundary + 1, len(tokens)):
                    local_reward_weight[i] = sample_weight

        local_syntax_weight = None
        if syntax_lookup is not None:
            local_syntax_weight = []
            for token_id in tokens:
                if 0 <= token_id < len(syntax_lookup):
                    local_syntax_weight.append(float(syntax_lookup[token_id]))
                else:
                    local_syntax_weight.append(1.0)

        if len(tokens) > max_tokens:
            truncated_sequences += 1
            tokens = tokens[:max_tokens]
            local_mask = local_mask[:max_tokens]
            if local_reward_weight is not None:
                local_reward_weight = local_reward_weight[:max_tokens]
            if local_syntax_weight is not None:
                local_syntax_weight = local_syntax_weight[:max_tokens]
            if boundary >= max_tokens:
                boundary = -1

        sep_cost = 1 if current_buf else 0
        if current_buf and len(current_buf) + sep_cost + len(tokens) > max_tokens:
            packed_rows.append(_finalize_pack(current_buf, max_tokens, pad_token))
            mask_rows.append(_finalize_pack(current_mask, max_tokens, 0))
            boundary_rows.append(current_bounds)
            if sample_metadata is not None:
                reward_weight_rows.append(_finalize_pack(current_reward_weight, max_tokens, 0.0, dtype=np.float16))
            if syntax_lookup is not None:
                syntax_weight_rows.append(_finalize_pack(current_syntax_weight, max_tokens, 1.0, dtype=np.float16))
            current_buf, current_mask, current_bounds = [], [], []
            current_reward_weight, current_syntax_weight = [], []

        offset = len(current_buf) + (1 if current_buf else 0)
        if current_buf:
            current_buf.append(SEPARATOR_TOKEN)
            current_mask.append(0) # Separator is not a target
            if sample_metadata is not None:
                current_reward_weight.append(0.0)
            if syntax_lookup is not None:
                current_syntax_weight.append(1.0)

        if boundary >= 0 and len(current_bounds) < MAX_SEQS_PER_PACK:
            current_bounds.append(offset + boundary)

        current_buf.extend(tokens)
        current_mask.extend(local_mask)
        if sample_metadata is not None and local_reward_weight is not None:
            current_reward_weight.extend(local_reward_weight)
        if syntax_lookup is not None and local_syntax_weight is not None:
            current_syntax_weight.extend(local_syntax_weight)

    if current_buf:
        packed_rows.append(_finalize_pack(current_buf, max_tokens, pad_token))
        mask_rows.append(_finalize_pack(current_mask, max_tokens, 0))
        boundary_rows.append(current_bounds)
        if sample_metadata is not None:
            reward_weight_rows.append(_finalize_pack(current_reward_weight, max_tokens, 0.0, dtype=np.float16))
        if syntax_lookup is not None:
            syntax_weight_rows.append(_finalize_pack(current_syntax_weight, max_tokens, 1.0, dtype=np.float16))

    packed_ids = np.array(packed_rows, dtype=np.int32)
    packed_masks = np.array(mask_rows, dtype=np.uint8)
    boundaries = np.full((len(boundary_rows), MAX_SEQS_PER_PACK), -1, dtype=np.int32)
    for i, bounds in enumerate(boundary_rows):
        for j, b in enumerate(bounds[:MAX_SEQS_PER_PACK]):
            boundaries[i, j] = b

    stats = {
        "input_sequences": input_sequences,
        "marker_hit_sequences": marker_hit_sequences,
        "truncated_sequences": truncated_sequences,
        "output_packs": len(packed_rows),
        "marker_hit_rate": (marker_hit_sequences / input_sequences) if input_sequences else 0.0,
    }
    reward_weights = np.array(reward_weight_rows, dtype=np.float16) if reward_weight_rows else None
    syntax_weights = np.array(syntax_weight_rows, dtype=np.float16) if syntax_weight_rows else None
    return packed_ids, packed_masks, boundaries, reward_weights, syntax_weights, stats


def _finalize_pack(buf: list[int] | list[float], max_tokens: int, pad_token: int | float, dtype=np.int32) -> np.ndarray:
    arr = np.array(buf[:max_tokens], dtype=dtype)
    if len(arr) < max_tokens:
        arr = np.pad(arr, (0, max_tokens - len(arr)), constant_values=pad_token)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack pre-tokenized dataset into fixed-length buffers.")
    parser.add_argument("--input", type=str, required=True, help="Path to .npy token array.")
    parser.add_argument("--output", type=str, required=True, help="Path to write packed .npy (masks and boundaries alongside).")
    parser.add_argument("--max-tokens", type=int, default=1536, help="Max tokens per packed buffer.")
    parser.add_argument("--assistant-token", type=int, required=True, help="Token ID for start-of-assistant turn (e.g., 4368 for Gemma 4). Must match training config assistant_id.")
    parser.add_argument("--pad-token", type=int, default=0, help="Padding token ID.")
    parser.add_argument("--metadata-jsonl", type=str, help="Optional JSONL dataset path used to derive packed reward weights from metadata.sample_weight.")
    parser.add_argument(
        "--pretokenize-audit",
        type=str,
        help="Audit JSON produced by pretokenize_dataset.py; used to align metadata when long rows were skipped.",
    )
    parser.add_argument("--prompt-contract-version", default="tikz_partial_decode_v1")
    parser.add_argument("--normalization-config-hash", default="")
    parser.add_argument("--disabled-rules", default="", help="Comma-separated disabled normalization/filter rules.")
    parser.add_argument(
        "--scoring-status",
        default=None,
        help="Scoring provenance marker. Defaults to skipped_plain_ce when --metadata-jsonl is omitted.",
    )
    parser.add_argument("--model-id", type=str, help="Optional model id. Required when --emit-syntax-weights is set.")
    parser.add_argument("--emit-syntax-weights", action="store_true", help="Emit packed syntax weights alongside ids/masks.")
    parser.add_argument("--syntax-structural-weight", type=float, default=5.0)
    parser.add_argument("--syntax-command-weight", type=float, default=2.0)
    parser.add_argument("--syntax-coordinate-weight", type=float, default=1.0)
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    masks_path = output_path.with_name(output_path.stem + "_masks.npy")
    boundaries_path = output_path.with_name(output_path.stem + "_boundaries.npy")
    reward_weights_path = output_path.with_name(output_path.stem + "_reward_weights.npy")
    syntax_weights_path = output_path.with_name(output_path.stem + "_syntax_weights.npy")
    audit_path = output_path.with_name(output_path.stem + "_audit.json")

    print(f"Loading {args.input}...")
    tokenized = np.load(args.input, allow_pickle=True)
    sample_metadata = _load_record_metadata(pathlib.Path(args.metadata_jsonl)) if args.metadata_jsonl else None
    pretokenize_audit_path = pathlib.Path(args.pretokenize_audit) if args.pretokenize_audit else input_path.with_name(input_path.stem + "_audit.json")
    pretokenize_audit_payload = None
    if pretokenize_audit_path.exists():
        pretokenize_audit_payload = json.loads(pretokenize_audit_path.read_text(encoding="utf-8"))
    syntax_lookup = None
    if args.emit_syntax_weights:
        if not args.model_id:
            raise RuntimeError("--model-id is required when --emit-syntax-weights is set.")
        syntax_lookup = _build_syntax_lookup(
            args.model_id,
            structural_weight=args.syntax_structural_weight,
            command_weight=args.syntax_command_weight,
            coordinate_weight=args.syntax_coordinate_weight,
        )

    packed_ids, packed_masks, boundaries, reward_weights, syntax_weights, stats = pack_dataset(
        tokenized,
        args.max_tokens,
        args.assistant_token,
        sample_metadata=sample_metadata,
        pretokenize_audit=pretokenize_audit_payload,
        syntax_lookup=syntax_lookup,
        pad_token=args.pad_token,
    )

    print(f"\nPacked {len(tokenized)} → {len(packed_ids)} rows.")
    print(f"Saving to {output_path}, {masks_path}, {boundaries_path}...")
    np.save(output_path, packed_ids)
    np.save(masks_path, packed_masks)
    np.save(boundaries_path, boundaries)
    if reward_weights is not None:
        np.save(reward_weights_path, reward_weights)
    if syntax_weights is not None:
        np.save(syntax_weights_path, syntax_weights)
    mask_zero_fraction = float(np.mean(packed_masks == 0)) if packed_masks.size else 0.0
    hash_inputs = [output_path, masks_path, boundaries_path]
    if reward_weights is not None:
        hash_inputs.append(reward_weights_path)
    if syntax_weights is not None:
        hash_inputs.append(syntax_weights_path)
    metadata_jsonl_path = pathlib.Path(args.metadata_jsonl).expanduser().resolve() if args.metadata_jsonl else None
    scoring_status = args.scoring_status or ("metadata_weighted" if metadata_jsonl_path else "skipped_plain_ce")
    audit = {
        "marker_hit_rate": float(stats["marker_hit_rate"]),
        "mask_zero_fraction": mask_zero_fraction,
        "assistant_token_used": int(args.assistant_token),
        "sequences_input": int(stats["input_sequences"]),
        "sequences_packed": int(stats["output_packs"]),
        "truncated_sequences": int(stats["truncated_sequences"]),
        "reward_weighted": reward_weights is not None,
        "syntax_weighted": syntax_weights is not None,
        "metadata_jsonl": str(metadata_jsonl_path) if metadata_jsonl_path is not None else None,
        "scoring_status": scoring_status,
        "prompt_contract_version": args.prompt_contract_version,
        "normalization_config_hash": args.normalization_config_hash,
        "disabled_rules": [value for value in args.disabled_rules.split(",") if value],
        "pretokenize_audit_path": str(pretokenize_audit_path.expanduser().resolve()) if pretokenize_audit_path.exists() else None,
        "pretokenize_audit_sha256": _file_sha256(pretokenize_audit_path) if pretokenize_audit_path.exists() else None,
        "pretokenize_source_jsonl_sha256": (
            pretokenize_audit_payload.get("source_jsonl_sha256")
            if isinstance(pretokenize_audit_payload, dict)
            else None
        ),
        "pretokenize_source_row_count": (
            pretokenize_audit_payload.get("source_row_count")
            if isinstance(pretokenize_audit_payload, dict)
            else None
        ),
        "source_dataset_sha256": _file_sha256(input_path),
        "dataset_sha256": _pack_dataset_sha256(*hash_inputs),
        "audit_timestamp": np.datetime_as_string(np.datetime64("now"), timezone="UTC"),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "Audit:"
        f" marker_hit_rate={audit['marker_hit_rate']:.3f},"
        f" mask_zero_fraction={audit['mask_zero_fraction']:.3f},"
        f" truncated_sequences={audit['truncated_sequences']},"
        f" audit_path={audit_path}"
    )
    print("Done.")



if __name__ == "__main__":
    main()
