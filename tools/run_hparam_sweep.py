#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _parse_csv(value: str, cast):
    items = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            items.append(cast(raw))
    return items


def _parse_layers(value: str) -> list[int | None]:
    layers: list[int | None] = []
    for raw in value.split(","):
        raw = raw.strip().lower()
        if not raw:
            continue
        if raw in {"all", "none", "null"}:
            layers.append(None)
        else:
            layers.append(int(raw))
    return layers


def _variant_id(
    stage: int,
    rank: int,
    lr: float,
    dropout: float,
    clip: float,
    weight_decay: float,
    layers: int | None,
) -> str:
    layer_text = "all" if layers is None else str(layers)
    return (
        f"stage{stage}_r{rank}_lr{lr:.1e}_drop{dropout:g}_clip{clip:g}_wd{weight_decay:g}_layers{layer_text}"
        .replace("+", "")
        .replace(".", "p")
    )


def _resolve_source_path(source_root: Path, value: Any) -> Any:
    if value in (None, ""):
        return value
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = source_root / path
    return str(path.resolve())


def _absolutize_training_paths(training: dict[str, Any], source_root: Path) -> None:
    path_keys = (
        "dataset_path",
        "train_dataset_path",
        "pretokenized_cache_path",
        "pretokenized_packed_cache_path",
        "val_dataset_path",
        "gold_eval_dataset_path",
        "reward_weight_path",
        "syntax_weight_path",
        "resume_adapter_path",
    )
    for key in path_keys:
        if key in training:
            training[key] = _resolve_source_path(source_root, training[key])

    stage2 = training.get("stage2")
    if isinstance(stage2, dict):
        stage2_path_keys = (
            "dataset_path",
            "val_dataset_path",
            "gold_eval_dataset_path",
            "resume_adapter_path",
        )
        for key in stage2_path_keys:
            if key in stage2:
                stage2[key] = _resolve_source_path(source_root, stage2[key])


def _absolutize_project_paths(config: dict[str, Any], *, source_root: Path, output_root: Path, runs_dir: Path) -> None:
    paths = config.setdefault("paths", {})
    for key in ("data_dir", "prepared_dir", "manifests_dir"):
        if key in paths:
            paths[key] = _resolve_source_path(source_root, paths[key])
    paths["outputs_dir"] = str((output_root / "outputs").resolve())
    paths["runs_dir"] = str(runs_dir.resolve())
    if "cache_dir" in paths:
        paths["cache_dir"] = str((output_root / "cache").resolve())

    compiler = config.get("compiler")
    if isinstance(compiler, dict) and "tectonic_binary" in compiler:
        compiler["tectonic_binary"] = _resolve_source_path(source_root, compiler["tectonic_binary"])

    training = config.get("training")
    if isinstance(training, dict):
        _absolutize_training_paths(training, source_root)


def _write_variant_config(base: dict[str, Any], path: Path, params: dict[str, Any]) -> None:
    config = json.loads(json.dumps(base))
    _absolutize_project_paths(
        config,
        source_root=Path(params["source_root"]),
        output_root=Path(params["output_root"]),
        runs_dir=Path(params["runs_dir"]),
    )
    training = config.setdefault("training", {})
    memory = config.setdefault("memory", {})
    training["lora_rank"] = params["rank"]
    training["lora_alpha"] = params["rank"] * 2
    training["lora_dropout"] = params["dropout"]
    training["learning_rate"] = params["learning_rate"]
    training["max_grad_norm"] = params["max_grad_norm"]
    training["weight_decay"] = params["weight_decay"]
    training["lora_num_layers"] = params["lora_num_layers"]
    training["iters"] = params["iters"]
    training["steps_per_save"] = params["save_interval"]
    training.setdefault("coverage", {})["enabled"] = True
    memory.setdefault("batch_size", 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _materialize_variant_files(base: dict[str, Any], variant: dict[str, Any]) -> None:
    """Recreate generated sweep files/directories that are safe to materialize."""
    config_path = Path(variant["config_path"])
    _write_variant_config(base, config_path, variant["params"])
    Path(variant["adapter_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(variant["gate_dir"]).mkdir(parents=True, exist_ok=True)
    Path(variant["params"]["runs_dir"], variant["run_id"]).mkdir(parents=True, exist_ok=True)


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_running_marker(output_root: Path) -> Path:
    marker = output_root / ".RUNNING.json"
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        pid = payload.get("pid")
        if isinstance(pid, int) and _pid_is_running(pid):
            raise RuntimeError(f"sweep output root is already active under pid {pid}: {output_root}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "output_root": str(output_root.resolve()),
                "argv": sys.argv,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def _finish_running_marker(marker: Path, exit_code: int) -> None:
    if not marker.exists():
        return
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload["exit_code"] = exit_code
    finished = marker.with_name(".FINISHED.json")
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(marker, finished)


def _variant_rank_score(gate_dir: Path) -> tuple[float, float, float, float]:
    results_path = gate_dir / "eval" / "results.json"
    if not results_path.exists():
        return (-1.0, -1.0, -1.0, -999.0)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    metrics = data.get("finetuned", {})
    return (
        -float(metrics.get("repetition_loop_rate", 1.0)),
        float(metrics.get("compile_rate", 0.0)),
        float(metrics.get("substantive_rate", 0.0)),
        -abs(1.0 - float(metrics.get("avg_code_length_ratio_vs_base", 99.0))),
    )


def _jsonl_has_row_aligned_example_index(path: Path) -> bool:
    if not path.exists():
        return False
    found = False
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            found = True
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return False
            if record.get("example_index") != row_index:
                return False
            metadata = record.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("example_index") != row_index:
                return False
    return found


def _find_raw_stage_dataset(source_root: Path, stage: int) -> Path | None:
    for candidate in (
        source_root / f"data/prepared/curriculum/train_stage{stage}.jsonl",
        source_root / f"data/prepared/train_stage{stage}.jsonl",
        source_root / "data/prepared/train.jsonl",
    ):
        if candidate.exists():
            return candidate
    return None


def _run_data_gates_if_needed(
    *,
    project_root: Path,
    source_root: Path,
    stage: int,
    base_config: Path,
    base: dict[str, Any],
    force: bool,
    skip: bool,
    env: dict[str, str],
) -> int:
    training = base.get("training", {})
    clean_jsonl = Path(_resolve_source_path(source_root, training.get("dataset_path")))
    pretok_raw = training.get("pretokenized_cache_path")
    pretok_path = Path(_resolve_source_path(source_root, pretok_raw)) if pretok_raw else None

    if skip:
        if not _jsonl_has_row_aligned_example_index(clean_jsonl):
            print(
                "ERROR: --skip-data-gates was used, but the clean dataset is missing row-aligned example_index: "
                f"{clean_jsonl}",
                file=sys.stderr,
            )
            return 1
        return 0

    if not force and _jsonl_has_row_aligned_example_index(clean_jsonl) and (pretok_path is None or pretok_path.exists()):
        print(f"Data gates already satisfied: {clean_jsonl}")
        return 0

    raw_dataset = _find_raw_stage_dataset(source_root, stage)
    if raw_dataset is None:
        print(
            f"ERROR: no raw training dataset found for stage {stage}; cannot regenerate {clean_jsonl}",
            file=sys.stderr,
        )
        return 1
    if pretok_path is None:
        print("ERROR: base config has no training.pretokenized_cache_path; sweep requires unpacked token cache.", file=sys.stderr)
        return 1

    cmd = [
        "bash",
        "tools/run_data_gates.sh",
        "--stage",
        str(stage),
        "--input",
        str(raw_dataset),
        "--clean-output",
        str(clean_jsonl),
        "--pretok-output",
        str(pretok_path),
    ]
    if stage > 1:
        cmd.append("--repair-contract")
    val_path = training.get("val_dataset_path")
    if val_path:
        cmd.extend(["--val", _resolve_source_path(source_root, val_path)])
    gold_path = training.get("gold_eval_dataset_path")
    if gold_path:
        cmd.extend(["--gold", _resolve_source_path(source_root, gold_path)])

    print(f"Running data gates before sweep: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root, env=env)
    if result.returncode != 0:
        return result.returncode
    if not _jsonl_has_row_aligned_example_index(clean_jsonl):
        print(f"ERROR: data gates completed but example_index validation still failed: {clean_jsonl}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and optionally execute 800-step LoRA hyperparameter/layer-count sweep trials."
    )
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--save-interval", type=int, default=400)
    parser.add_argument("--ranks", default="16,24,32")
    parser.add_argument("--learning-rates", default="4e-6,2e-6")
    parser.add_argument("--dropouts", default="0.03,0.05")
    parser.add_argument("--max-grad-norms", default="0.5,1.0")
    parser.add_argument("--weight-decays", default="0.01,0.03")
    parser.add_argument("--lora-num-layers", default="28,all")
    parser.add_argument("--resume-adapter", default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--gate-mode", choices=("quick", "full", "promote"), default="quick")
    parser.add_argument("--quick-num-samples", type=int, default=32)
    parser.add_argument("--promote-top-k", type=int, default=3)
    parser.add_argument("--promote-num-samples", type=int, default=100)
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for smoke-testing a subset.")
    parser.add_argument("--execute", action="store_true", help="Actually run the generated trial commands.")
    parser.add_argument("--skip-stage-gate", action="store_true")
    parser.add_argument("--skip-data-gates", action="store_true", help="Use existing clean/tokenized data after validating example_index.")
    parser.add_argument("--force-data-gates", action="store_true", help="Regenerate clean/tokenized data before executing.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_config = Path(args.base_config or f"configs/curriculum_stage{args.stage}.yaml")
    if not base_config.exists():
        print(f"ERROR: base config does not exist: {base_config}", file=sys.stderr)
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or f"outputs/hparam_sweeps/stage{args.stage}_{timestamp}")
    configs_dir = output_root / "configs"
    adapters_dir = output_root / "adapters"
    runs_dir = output_root / "runs"
    gates_dir = output_root / "gates"
    base_cache_dir = output_root / "base_eval_cache"
    manifest_path = output_root / "manifest.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)

    with base_config.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    source_root = base_config.resolve().parent.parent
    grad_accum = int(base.get("memory", {}).get("gradient_accumulation_steps", 1))
    if grad_accum < 1:
        print(f"ERROR: gradient_accumulation_steps must be >= 1, got {grad_accum}", file=sys.stderr)
        return 1
    if args.iters % grad_accum != 0:
        print(
            f"ERROR: --iters must be divisible by gradient_accumulation_steps={grad_accum} "
            f"for strict no-repeat coverage.",
            file=sys.stderr,
        )
        return 1
    if args.save_interval % grad_accum != 0:
        print(
            f"ERROR: --save-interval must be divisible by gradient_accumulation_steps={grad_accum}.",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root / 'src'}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(project_root / "src")

    running_marker: Path | None = None
    if args.execute:
        try:
            running_marker = _start_running_marker(output_root)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    exit_code = 0
    try:
        if args.execute:
            data_gate_status = _run_data_gates_if_needed(
                project_root=project_root,
                source_root=source_root,
                stage=args.stage,
                base_config=base_config,
                base=base,
                force=args.force_data_gates,
                skip=args.skip_data_gates,
                env=env,
            )
            if data_gate_status != 0:
                exit_code = data_gate_status
                return exit_code
        else:
            clean_jsonl = Path(_resolve_source_path(source_root, base.get("training", {}).get("dataset_path")))
            if not _jsonl_has_row_aligned_example_index(clean_jsonl):
                print(
                    f"WARNING: {clean_jsonl} is missing row-aligned example_index. "
                    "An executing sweep will run data gates before training."
                )

        exit_code = _run_sweep(args, base, project_root, output_root, configs_dir, adapters_dir, runs_dir, gates_dir, base_cache_dir, manifest_path, source_root, env)
        return exit_code
    finally:
        if running_marker is not None:
            marker_exit_code = 1 if sys.exc_info()[0] is not None else exit_code
            _finish_running_marker(running_marker, marker_exit_code)


def _run_sweep(
    args: argparse.Namespace,
    base: dict[str, Any],
    project_root: Path,
    output_root: Path,
    configs_dir: Path,
    adapters_dir: Path,
    runs_dir: Path,
    gates_dir: Path,
    base_cache_dir: Path,
    manifest_path: Path,
    source_root: Path,
    env: dict[str, str],
) -> int:
    ranks = _parse_csv(args.ranks, int)
    learning_rates = _parse_csv(args.learning_rates, float)
    dropouts = _parse_csv(args.dropouts, float)
    max_grad_norms = _parse_csv(args.max_grad_norms, float)
    weight_decays = _parse_csv(args.weight_decays, float)
    layer_counts = _parse_layers(args.lora_num_layers)

    variants: list[dict[str, Any]] = []
    for combo in itertools.product(ranks, learning_rates, dropouts, max_grad_norms, weight_decays, layer_counts):
        rank, lr, dropout, clip, wd, layers = combo
        variant_id = _variant_id(args.stage, rank, lr, dropout, clip, wd, layers)
        config_path = configs_dir / f"{variant_id}.yaml"
        adapter_path = adapters_dir / f"{variant_id}.safetensors"
        run_id = f"sweep_{variant_id}"
        params = {
            "rank": rank,
            "learning_rate": lr,
            "dropout": dropout,
            "max_grad_norm": clip,
            "weight_decay": wd,
            "lora_num_layers": layers,
            "iters": args.iters,
            "save_interval": args.save_interval,
            "source_root": str(source_root),
            "output_root": str(output_root.resolve()),
            "runs_dir": str(runs_dir.resolve()),
        }
        _write_variant_config(base, config_path, params)

        train_cmd = [
            sys.executable,
            "-u",
            "-m",
            "tikz_mlx.cli",
            "train",
            "--config",
            str(config_path),
            "--output-path",
            str(adapter_path),
            "--run-id",
            run_id,
            "--iters",
            str(args.iters),
            "--save-interval",
            str(args.save_interval),
            "--skip-post-ab-eval",
        ]
        if args.resume_adapter:
            train_cmd.extend(["--resume-adapter", args.resume_adapter])

        gate_samples = args.quick_num_samples if args.gate_mode == "quick" else args.num_samples
        quick_gate_dir = gates_dir / variant_id / args.gate_mode
        gate_cmd = [
            "bash",
            "tools/run_stage_gate.sh",
            "--config",
            str(config_path),
            "--adapter",
            str(adapter_path),
            "--checkpoint-dir",
            str(runs_dir / run_id),
            "--num-samples",
            str(gate_samples),
            "--gate-mode",
            args.gate_mode,
            "--out-dir",
            str(quick_gate_dir),
            "--base-cache-dir",
            str(base_cache_dir),
            "--compile-workers",
            str(args.compile_workers),
        ]
        variants.append(
            {
                "variant_id": variant_id,
                "config_path": str(config_path),
                "adapter_path": str(adapter_path),
                "run_id": run_id,
                "params": params,
                "train_cmd": train_cmd,
                "gate_cmd": gate_cmd,
                "gate_dir": str(quick_gate_dir),
            }
        )
        if args.max_runs is not None and len(variants) >= args.max_runs:
            break

    with manifest_path.open("w", encoding="utf-8") as handle:
        for variant in variants:
            handle.write(json.dumps(variant, sort_keys=True) + "\n")

    print(f"Wrote {len(variants)} sweep variants to {manifest_path}")
    if not args.execute:
        print("Dry plan only. Re-run with --execute to launch trials.")
        return 0

    training_failures = 0
    gate_failures = 0
    promote_failures = 0
    promote_passed = 0
    quick_passed: list[dict[str, Any]] = []
    for variant in variants:
        print(f"\n=== Running {variant['variant_id']} ===")
        _materialize_variant_files(base, variant)
        train_result = subprocess.run(variant["train_cmd"], cwd=project_root, env=env)
        if train_result.returncode != 0:
            training_failures += 1
            print(f"ERROR: training failed for {variant['variant_id']} with code {train_result.returncode}")
            continue
        if args.skip_stage_gate:
            continue
        _materialize_variant_files(base, variant)
        gate_result = subprocess.run(variant["gate_cmd"], cwd=project_root, env=env)
        if gate_result.returncode != 0:
            gate_failures += 1
            print(f"ERROR: stage gate failed for {variant['variant_id']} with code {gate_result.returncode}")
        else:
            quick_passed.append(variant)

    if (
        not args.skip_stage_gate
        and args.gate_mode == "quick"
        and args.promote_top_k > 0
        and quick_passed
    ):
        ranked = sorted(
            quick_passed,
            key=lambda item: _variant_rank_score(Path(item["gate_dir"])),
            reverse=True,
        )
        promote_variants = ranked[: args.promote_top_k]
        print(f"\n=== Running full promote gate for top {len(promote_variants)} quick-gate survivors ===")
        for variant in promote_variants:
            _materialize_variant_files(base, variant)
            promote_dir = gates_dir / variant["variant_id"] / "promote"
            promote_cmd = [
                "bash",
                "tools/run_stage_gate.sh",
                "--config",
                variant["config_path"],
                "--adapter",
                variant["adapter_path"],
                "--checkpoint-dir",
                str(runs_dir / variant["run_id"]),
                "--num-samples",
                str(args.promote_num_samples),
                "--gate-mode",
                "promote",
                "--out-dir",
                str(promote_dir),
                "--base-cache-dir",
                str(base_cache_dir),
                "--compile-workers",
                str(args.compile_workers),
            ]
            print(f"\n=== Promoting {variant['variant_id']} ===")
            promote_result = subprocess.run(promote_cmd, cwd=project_root, env=env)
            if promote_result.returncode != 0:
                promote_failures += 1
                print(f"ERROR: promote gate failed for {variant['variant_id']} with code {promote_result.returncode}")
            else:
                promote_passed += 1

    print(
        "\nSweep summary: "
        f"training_failures={training_failures}, "
        f"gate_failures={gate_failures}, "
        f"quick_gate_passes={len(quick_passed)}, "
        f"promote_passes={promote_passed}, "
        f"promote_failures={promote_failures}"
    )
    if training_failures:
        return 1
    if not args.skip_stage_gate and not quick_passed:
        return 1
    if args.gate_mode == "quick" and args.promote_top_k > 0 and quick_passed and promote_passed == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
