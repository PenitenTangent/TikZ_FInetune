#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tikz_mlx.recovery import (  # noqa: E402
    DEFAULT_MODE_CAPS,
    QualityFilterConfig,
    assert_manifest_sets_disjoint,
    build_eval_manifest,
    evaluate_ab_result_gate,
    filter_quality_records,
    has_repetition_failure,
    iter_jsonl,
    repetition_features,
    select_mode_capped_records,
    select_equal_mode_sample_ids,
    synthetic_repetition_examples,
    repair_assistant_contract,
    stability_checkpoint_from_dict,
    select_stability_checkpoint,
    validate_contract_file,
    validate_split_disjoint,
    validate_sweep_resume_adapter,
    write_gate_config,
    write_jsonl,
)


def _cmd_gate_config(args: argparse.Namespace) -> int:
    payload = write_gate_config(Path(args.output), overwrite=args.overwrite)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    payload = build_eval_manifest(
        Path(args.dataset),
        seed=args.seed,
        max_tokens=args.max_tokens,
        decoding_config={"profile": args.decoding_profile},
        compiler_config={"config": args.compiler_config},
    )
    assert_manifest_sets_disjoint(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest_path": str(output), "manifest_sha256": payload["manifest_sha256"]}, indent=2))
    return 0


def _cmd_split_check(args: argparse.Namespace) -> int:
    counts = validate_split_disjoint(
        {
            "train": Path(args.train),
            "val": Path(args.val),
            "gold": Path(args.gold),
        }
    )
    print(json.dumps({"counts": counts}, indent=2, sort_keys=True))
    return 0


def _cmd_contract_check(args: argparse.Namespace) -> int:
    payload = validate_contract_file(Path(args.dataset), limit=args.limit)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["failure_count"]:
        return 1
    return 0


def _cmd_filter_clean_data(args: argparse.Namespace) -> int:
    records = list(iter_jsonl(Path(args.input)))
    if args.repair_contract:
        records = [repair_assistant_contract(record)[0] for record in records]
    config = QualityFilterConfig(max_token_length=args.max_token_length)
    kept, audit = filter_quality_records(records, config=config)
    write_jsonl(Path(args.output), kept)
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": args.output, "audit": args.audit_output, **audit}, indent=2, sort_keys=True))
    return 0


def _cmd_build_clean_stage_split(args: argparse.Namespace) -> int:
    records = list(iter_jsonl(Path(args.input)))
    repair_audit: list[dict] = []
    if args.repair_contract:
        repaired_records = []
        for record in records:
            repaired, item_audit = repair_assistant_contract(record)
            repaired_records.append(repaired)
            if item_audit["changed"] or item_audit["original_violations"] or item_audit["repaired_violations"]:
                repair_audit.append(item_audit)
        records = repaired_records
    config = QualityFilterConfig(max_token_length=args.max_token_length)
    kept, audit = filter_quality_records(records, config=config)

    selected_probe_ids = set(
        select_equal_mode_sample_ids(
            kept,
            total=args.probe_size,
            seed=args.seed,
            min_remaining_per_mode=args.min_train_per_mode,
        )
    )
    probe_records: list[dict] = []
    train_records: list[dict] = []
    for index, record in enumerate(kept):
        sample_id = str(record.get("sample_id", f"row_{index:06d}"))
        if sample_id in selected_probe_ids:
            probe_records.append(record)
        else:
            train_records.append(record)

    write_jsonl(Path(args.train_output), train_records)
    write_jsonl(Path(args.probe_output), probe_records)

    split_audit = {
        **audit,
        "seed": args.seed,
        "probe_size_requested": args.probe_size,
        "min_train_per_mode": args.min_train_per_mode,
        "probe_records": len(probe_records),
        "train_records": len(train_records),
        "probe_sample_ids": [str(record.get("sample_id", "")) for record in probe_records],
        "probe_mode_counts": {},
        "train_mode_counts": {},
        "contract_repair_enabled": bool(args.repair_contract),
        "contract_repair_examples": repair_audit[:100],
        "contract_repair_changed": sum(1 for item in repair_audit if item["changed"]),
        "contract_repair_failed": sum(1 for item in repair_audit if item["repaired_violations"]),
    }
    for record in probe_records:
        mode = str((record.get("metadata") or {}).get("generation_mode", "unknown"))
        split_audit["probe_mode_counts"][mode] = split_audit["probe_mode_counts"].get(mode, 0) + 1
    for record in train_records:
        mode = str((record.get("metadata") or {}).get("generation_mode", "unknown"))
        split_audit["train_mode_counts"][mode] = split_audit["train_mode_counts"].get(mode, 0) + 1

    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "train_output": args.train_output,
                "probe_output": args.probe_output,
                "audit_output": args.audit_output,
                "train_records": len(train_records),
                "probe_records": len(probe_records),
                "total_rejected": audit["total_rejected"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_repair_contract(args: argparse.Namespace) -> int:
    repaired_records: list[dict] = []
    rejected_records: list[dict] = []
    audit_items: list[dict] = []
    for record in iter_jsonl(Path(args.input)):
        repaired, item_audit = repair_assistant_contract(record)
        if item_audit["repaired_violations"] and args.drop_failed:
            rejected = json.loads(json.dumps(record))
            rejected["_reject_reasons"] = [
                f"contract_repair:{item}" for item in item_audit["repaired_violations"]
            ]
            rejected_records.append(rejected)
        else:
            repaired_records.append(repaired)
        if item_audit["changed"] or item_audit["original_violations"] or item_audit["repaired_violations"]:
            audit_items.append(item_audit)
    write_jsonl(Path(args.output), repaired_records)
    if args.rejected_output:
        write_jsonl(Path(args.rejected_output), rejected_records)
    audit = {
        "input": args.input,
        "output": args.output,
        "records": len(repaired_records),
        "dropped_failed_repair": len(rejected_records),
        "changed": sum(1 for item in audit_items if item["changed"]),
        "failed_after_repair": sum(1 for item in audit_items if item["repaired_violations"]),
        "examples": audit_items[:100],
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.drop_failed:
        return 0
    return 0 if audit["failed_after_repair"] == 0 else 1


def _cmd_mode_balance(args: argparse.Namespace) -> int:
    records = list(iter_jsonl(Path(args.input)))
    caps = DEFAULT_MODE_CAPS
    if args.caps_json:
        raw_caps = json.loads(args.caps_json)
        if not isinstance(raw_caps, dict):
            raise RuntimeError("--caps-json must be a JSON object.")
        caps = {str(key): (None if value is None else int(value)) for key, value in raw_caps.items()}
    selected, audit = select_mode_capped_records(records, caps=caps, seed=args.seed)
    write_jsonl(Path(args.output), selected)
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": args.output, "audit": args.audit_output, **audit}, indent=2, sort_keys=True))
    return 0


def _cmd_repetition_sidecar(args: argparse.Namespace) -> int:
    examples = synthetic_repetition_examples()
    seen_texts = {str(example["text"]) for example in examples}
    for mine_dir in args.mine_dir:
        root = Path(mine_dir)
        if not root.exists():
            continue
        for path in sorted(root.rglob("raw_response.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if text in seen_texts or not has_repetition_failure(text):
                continue
            seen_texts.add(text)
            examples.append(
                {
                    "sample_id": f"mined_repetition_{len(examples):05d}",
                    "source": str(path),
                    "text": text,
                    "features": repetition_features(text),
                }
            )
            if len(examples) >= args.max_examples:
                break
        if len(examples) >= args.max_examples:
            break

    output = Path(args.output)
    write_jsonl(output, examples)
    failures = [
        str(example.get("sample_id"))
        for example in examples
        if not bool((example.get("features") or {}).get("has_repetition_loop"))
    ]
    if failures:
        raise RuntimeError(f"Repetition sidecar contains examples that do not trigger detector: {failures[:10]}")
    print(json.dumps({"output": str(output), "records": len(examples)}, indent=2, sort_keys=True))
    return 0


def _cmd_subset(args: argparse.Namespace) -> int:
    records = list(iter_jsonl(Path(args.input)))
    selected = set(select_equal_mode_sample_ids(records, total=args.size, seed=args.seed))
    subset = []
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id", f"row_{index:06d}"))
        if sample_id in selected:
            subset.append(record)
    write_jsonl(Path(args.output), subset)
    print(json.dumps({"output": args.output, "records": len(subset)}, indent=2, sort_keys=True))
    return 0


def _cmd_materialize_eval_set(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    sets = manifest.get("sets")
    if not isinstance(sets, dict) or args.set_name not in sets:
        raise RuntimeError(f"Eval manifest does not contain set: {args.set_name}")
    requested_ids = [str(value) for value in sets[args.set_name]]
    requested = set(requested_ids)
    records_by_id: dict[str, dict] = {}
    for index, record in enumerate(iter_jsonl(Path(args.dataset))):
        sample_id = str(record.get("sample_id", f"row_{index:06d}"))
        if sample_id in requested:
            records_by_id[sample_id] = record
    missing = [sample_id for sample_id in requested_ids if sample_id not in records_by_id]
    if missing:
        raise RuntimeError(f"Dataset is missing manifest sample_ids for {args.set_name}: {missing[:10]}")
    write_jsonl(Path(args.output), [records_by_id[sample_id] for sample_id in requested_ids])
    print(json.dumps({"output": args.output, "set": args.set_name, "records": len(requested_ids)}, indent=2))
    return 0


def _cmd_select_stability(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Stability input must be a JSON list of checkpoint metric objects.")
    selected = select_stability_checkpoint(
        [stability_checkpoint_from_dict(item) for item in payload],
        base_stability_emd=args.base_stability_emd,
    )
    print(json.dumps({"selected": selected.checkpoint_path, "iteration": selected.iteration}, indent=2))
    return 0


def _cmd_validate_resume(args: argparse.Namespace) -> int:
    validate_sweep_resume_adapter(
        Path(args.adapter_metadata) if args.adapter_metadata else None,
        Path(args.archive_manifest) if args.archive_manifest else None,
    )
    print(json.dumps({"valid": True}, indent=2))
    return 0


def _cmd_check_ab_gate(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("A/B results must be a JSON object.")
    gate_config = None
    if args.gate_config:
        gate_config_payload = json.loads(Path(args.gate_config).read_text(encoding="utf-8"))
        if not isinstance(gate_config_payload, dict):
            raise RuntimeError("Gate config must be a JSON object.")
        gate_config = {
            str(key): float(value)
            for key, value in gate_config_payload.items()
            if isinstance(value, (int, float))
        }
    result = evaluate_ab_result_gate(
        payload,
        candidate_key=args.candidate_key,
        base_key=args.base_key,
        gate=args.gate,
        gate_config=gate_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recovery-plan artifact and gate utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("write-gate-config")
    gate.add_argument("--output", default="data/manifests/gate_config_v1.json")
    gate.add_argument("--overwrite", action="store_true")
    gate.set_defaults(func=_cmd_gate_config)

    manifest = subparsers.add_parser("build-eval-manifest")
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--output", default="data/manifests/eval_manifest_v1.json")
    manifest.add_argument("--seed", type=int, default=20260427)
    manifest.add_argument("--max-tokens", type=int, default=2048)
    manifest.add_argument("--decoding-profile", default="recovery")
    manifest.add_argument("--compiler-config", default="config")
    manifest.set_defaults(func=_cmd_manifest)

    split = subparsers.add_parser("check-splits")
    split.add_argument("--train", required=True)
    split.add_argument("--val", required=True)
    split.add_argument("--gold", required=True)
    split.set_defaults(func=_cmd_split_check)

    contract = subparsers.add_parser("check-contract")
    contract.add_argument("--dataset", required=True)
    contract.add_argument("--limit", type=int)
    contract.set_defaults(func=_cmd_contract_check)

    clean = subparsers.add_parser("filter-clean-data")
    clean.add_argument("--input", required=True)
    clean.add_argument("--output", required=True)
    clean.add_argument("--audit-output", default="data/manifests/clean_data_filter_audit.json")
    clean.add_argument("--max-token-length", type=int, default=1536)
    clean.add_argument("--repair-contract", action="store_true")
    clean.set_defaults(func=_cmd_filter_clean_data)

    repair = subparsers.add_parser("repair-contract")
    repair.add_argument("--input", required=True)
    repair.add_argument("--output", required=True)
    repair.add_argument("--audit-output", required=True)
    repair.add_argument("--drop-failed", action="store_true")
    repair.add_argument("--rejected-output")
    repair.set_defaults(func=_cmd_repair_contract)

    clean_stage = subparsers.add_parser(
        "build-clean-stage-split",
        help="Filter a stage dataset and split out a fixed held-out stage-distribution probe.",
    )
    clean_stage.add_argument("--input", required=True)
    clean_stage.add_argument("--train-output", required=True)
    clean_stage.add_argument("--probe-output", required=True)
    clean_stage.add_argument("--audit-output", required=True)
    clean_stage.add_argument("--probe-size", type=int, default=50)
    clean_stage.add_argument("--max-token-length", type=int, required=True)
    clean_stage.add_argument("--seed", type=int, default=42)
    clean_stage.add_argument(
        "--min-train-per-mode",
        type=int,
        default=0,
        help="Reserve at least this many clean records per mode when selecting the held-out probe.",
    )
    clean_stage.add_argument("--repair-contract", action="store_true")
    clean_stage.set_defaults(func=_cmd_build_clean_stage_split)

    balance = subparsers.add_parser("build-mode-balanced-train")
    balance.add_argument("--input", required=True)
    balance.add_argument("--output", required=True)
    balance.add_argument("--audit-output", default="data/manifests/mode_balance_audit.json")
    balance.add_argument("--seed", type=int, default=20260427)
    balance.add_argument("--caps-json")
    balance.set_defaults(func=_cmd_mode_balance)

    sidecar = subparsers.add_parser("build-repetition-sidecar")
    sidecar.add_argument("--output", default="data/prepared/repetition_penalty_examples.jsonl")
    sidecar.add_argument("--mine-dir", action="append", default=["outputs"])
    sidecar.add_argument("--max-examples", type=int, default=200)
    sidecar.set_defaults(func=_cmd_repetition_sidecar)

    subset = subparsers.add_parser("build-stratified-subset")
    subset.add_argument("--input", required=True)
    subset.add_argument("--output", required=True)
    subset.add_argument("--size", type=int, required=True)
    subset.add_argument("--seed", type=int, default=20260427)
    subset.set_defaults(func=_cmd_subset)

    materialize = subparsers.add_parser("materialize-eval-set")
    materialize.add_argument("--dataset", required=True)
    materialize.add_argument("--manifest", required=True)
    materialize.add_argument("--set-name", required=True)
    materialize.add_argument("--output", required=True)
    materialize.set_defaults(func=_cmd_materialize_eval_set)

    stability = subparsers.add_parser("select-stability")
    stability.add_argument("--input", required=True)
    stability.add_argument("--base-stability-emd", type=float, required=True)
    stability.set_defaults(func=_cmd_select_stability)

    resume = subparsers.add_parser("validate-sweep-resume")
    resume.add_argument("--adapter-metadata")
    resume.add_argument("--archive-manifest")
    resume.set_defaults(func=_cmd_validate_resume)

    ab_gate = subparsers.add_parser("check-ab-gate")
    ab_gate.add_argument("--results", required=True)
    ab_gate.add_argument("--candidate-key", required=True)
    ab_gate.add_argument("--base-key", default="base")
    ab_gate.add_argument("--gate", choices=("sentinel", "ablation", "promotion"), required=True)
    ab_gate.add_argument("--gate-config")
    ab_gate.set_defaults(func=_cmd_check_ab_gate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
