from __future__ import annotations

import json
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpointing import (
    CheckpointContext,
    build_canonical_checkpoint_metadata,
    link_or_copy_atomic,
    utc_now_iso8601,
    write_checkpoint_metadata,
)
from .schema_validation import is_canonical_tikz_document as _is_canonical_tikz_document


@dataclass(slots=True)
class PromotionMetrics:
    label: str
    compile_rate: float
    schema_rate: float
    schema_samples: int
    mean_emd: float | None = None
    hybrid_pass_rate: float | None = None
    repetition_loop_rate: float | None = None
    truncation_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "compile_rate": self.compile_rate,
            "schema_rate": self.schema_rate,
            "schema_samples": self.schema_samples,
            "mean_emd": self.mean_emd,
            "hybrid_pass_rate": self.hybrid_pass_rate,
            "repetition_loop_rate": self.repetition_loop_rate,
            "truncation_rate": self.truncation_rate,
        }


def is_canonical_tikz_document(text: str) -> bool:
    return _is_canonical_tikz_document(text)


def _resolve_code_from_prompt_record(record: dict[str, Any]) -> str | None:
    inline_candidates = (
        record.get("code"),
        record.get("generated_code"),
        record.get("candidate_code"),
        record.get("latex"),
    )
    for candidate in inline_candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    tex_path_value = record.get("tex_path")
    if isinstance(tex_path_value, str) and tex_path_value.strip():
        tex_path = Path(tex_path_value)
        if tex_path.exists():
            return tex_path.read_text(encoding="utf-8")

    log_path_value = record.get("log_path")
    if isinstance(log_path_value, str) and log_path_value.strip():
        candidate_tex = Path(log_path_value).with_suffix(".tex")
        if candidate_tex.exists():
            return candidate_tex.read_text(encoding="utf-8")

    working_dir_value = record.get("working_dir")
    if isinstance(working_dir_value, str) and working_dir_value.strip():
        candidate_tex = Path(working_dir_value) / "candidate.tex"
        if candidate_tex.exists():
            return candidate_tex.read_text(encoding="utf-8")

    return None


def _extract_compile_rate(payload: dict[str, Any]) -> float | None:
    direct_candidates = (
        payload.get("compile_rate"),
        payload.get("compilation_rate"),
        payload.get("substantive_compile_success_rate"),
        payload.get("substantive_mark_mean"),
    )
    for candidate in direct_candidates:
        if isinstance(candidate, (int, float)):
            return float(candidate)

    compile_success = payload.get("compile_success")
    total = payload.get("total")
    if isinstance(compile_success, int) and isinstance(total, int) and total > 0:
        return float(compile_success) / float(total)
    return None


def _extract_schema_rate(payload: dict[str, Any]) -> tuple[float | None, int]:
    schema_rate_source = payload.get("schema_rate_source")
    schema_rate = payload.get("schema_rate")
    if isinstance(schema_rate_source, str) and isinstance(schema_rate, (int, float)):
        return float(schema_rate), 0

    substantive_rate = payload.get("substantive_tikz_rate")
    if isinstance(substantive_rate, (int, float)):
        return float(substantive_rate), 0

    per_prompt = payload.get("per_prompt")
    if not isinstance(per_prompt, list):
        return None, 0

    checked = 0
    schema_ok = 0
    for entry in per_prompt:
        if not isinstance(entry, dict):
            continue
        code = _resolve_code_from_prompt_record(entry)
        if code is None:
            continue
        checked += 1
        if is_canonical_tikz_document(code):
            schema_ok += 1

    if checked == 0:
        return None, 0
    return schema_ok / checked, checked


def _extract_optional_float(payload: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _resolve_result_block(payload: dict[str, Any], key: str | None) -> tuple[str, dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, dict):
        if key is not None:
            selected = payload.get(key)
            if not isinstance(selected, dict):
                raise RuntimeError(f"Result key '{key}' was not found in report results.")
            return key, selected
        return "report", payload

    if key is not None:
        if key not in results:
            raise RuntimeError(f"Result key '{key}' was not found in report results.")
        result = results[key]
        if not isinstance(result, dict):
            raise RuntimeError(f"Result block for key '{key}' is not a JSON object.")
        return key, result

    if len(results) != 1:
        available_keys = ", ".join(sorted(results.keys()))
        raise RuntimeError(
            "Report contains multiple result blocks; provide --baseline-key/--candidate-key. "
            f"Available keys: {available_keys}"
        )

    selected_key = next(iter(results))
    selected_block = results[selected_key]
    if not isinstance(selected_block, dict):
        raise RuntimeError(f"Result block for key '{selected_key}' is not a JSON object.")
    return selected_key, selected_block


def load_promotion_metrics(
    *,
    report_path: Path,
    result_key: str | None,
    compile_rate_override: float | None,
) -> PromotionMetrics:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Promotion report is not a JSON object: {report_path}")

    label, result_block = _resolve_result_block(payload, result_key)

    compile_rate = (
        float(compile_rate_override)
        if compile_rate_override is not None
        else _extract_compile_rate(result_block)
    )
    if compile_rate is None:
        raise RuntimeError(
            "Could not determine compile rate from report. Provide --baseline-compile-rate/--candidate-compile-rate."
        )

    schema_rate, schema_samples = _extract_schema_rate(result_block)
    if schema_rate is None:
        raise RuntimeError(
            "Could not determine strict schema adherence rate from report. "
            "Include code-bearing per_prompt entries (inline code, tex_path, log_path, or working_dir)."
        )

    return PromotionMetrics(
        label=label,
        compile_rate=compile_rate,
        schema_rate=schema_rate,
        schema_samples=schema_samples,
        mean_emd=_extract_optional_float(
            result_block,
            ("mean_emd", "average_emd", "emd_distance", "phase_b_emd", "visual_emd"),
        ),
        hybrid_pass_rate=_extract_optional_float(
            result_block,
            ("hybrid_pass_rate", "hybrid_visual_pass_rate", "hybrid_rate"),
        ),
        repetition_loop_rate=_extract_optional_float(
            result_block,
            ("repetition_loop_rate", "repetition_rate", "loop_rate"),
        ),
        truncation_rate=_extract_optional_float(
            result_block,
            ("truncation_rate", "truncated_rate"),
        ),
    )


def load_gate_config(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Gate config is not a JSON object: {path}")
    result: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


def evaluate_sft_promotion_gate(
    *,
    baseline: PromotionMetrics,
    candidate: PromotionMetrics,
    min_compile_delta: float,
    min_schema_delta: float,
    min_candidate_compile_rate: float,
    min_candidate_schema_rate: float,
    gate_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    gate_config = gate_config or {}
    if gate_config:
        configured_floor = gate_config.get("promotion_min_compile_rate")
        if configured_floor is not None:
            min_candidate_compile_rate = configured_floor
        min_compile_delta = max(min_compile_delta, 0.0)
        min_schema_delta = max(min_schema_delta, 0.0)

    compile_delta = candidate.compile_rate - baseline.compile_rate
    schema_delta = candidate.schema_rate - baseline.schema_rate

    checks = {
        "compile_delta": {
            "required": min_compile_delta,
            "observed": compile_delta,
            "passed": compile_delta >= min_compile_delta,
        },
        "schema_delta": {
            "required": min_schema_delta,
            "observed": schema_delta,
            "passed": schema_delta >= min_schema_delta,
        },
        "candidate_compile_floor": {
            "required": min_candidate_compile_rate,
            "observed": candidate.compile_rate,
            "passed": candidate.compile_rate >= min_candidate_compile_rate,
        },
        "candidate_schema_floor": {
            "required": min_candidate_schema_rate,
            "observed": candidate.schema_rate,
            "passed": candidate.schema_rate >= min_candidate_schema_rate,
        },
    }
    if gate_config:
        checks["candidate_compile_vs_base"] = {
            "required": baseline.compile_rate,
            "observed": candidate.compile_rate,
            "passed": candidate.compile_rate >= baseline.compile_rate,
        }
        checks["candidate_substantive_vs_base"] = {
            "required": baseline.schema_rate,
            "observed": candidate.schema_rate,
            "passed": candidate.schema_rate >= baseline.schema_rate,
        }
        if baseline.mean_emd is not None and candidate.mean_emd is not None:
            checks["candidate_mean_emd_vs_base"] = {
                "required": baseline.mean_emd,
                "observed": candidate.mean_emd,
                "passed": candidate.mean_emd <= baseline.mean_emd,
            }
        if baseline.hybrid_pass_rate is not None and candidate.hybrid_pass_rate is not None:
            hybrid_floor = max(
                gate_config.get("hybrid_visual_score_threshold", 0.75),
                baseline.hybrid_pass_rate,
            )
            checks["candidate_hybrid_pass_floor"] = {
                "required": hybrid_floor,
                "observed": candidate.hybrid_pass_rate,
                "passed": candidate.hybrid_pass_rate >= hybrid_floor,
            }
        repetition_ceiling = gate_config.get("repetition_loop_rate_max")
        if repetition_ceiling is not None:
            checks["candidate_repetition_loop_ceiling"] = {
                "required": repetition_ceiling,
                "observed": candidate.repetition_loop_rate,
                "passed": candidate.repetition_loop_rate is not None
                and candidate.repetition_loop_rate <= repetition_ceiling,
            }
        truncation_ceiling = gate_config.get("truncation_rate_max")
        if truncation_ceiling is not None:
            checks["candidate_truncation_ceiling"] = {
                "required": truncation_ceiling,
                "observed": candidate.truncation_rate,
                "passed": candidate.truncation_rate is not None
                and candidate.truncation_rate <= truncation_ceiling,
            }

    passed = all(item["passed"] for item in checks.values())
    return {
        "passed": passed,
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
        "checks": checks,
    }


def promote_sft_checkpoint(
    *,
    candidate_checkpoint_path: Path,
    sft_final_path: Path,
    policy_init_path: Path,
    force_policy_init: bool,
    run_id: str,
    gate_result: dict[str, Any],
) -> dict[str, Any]:
    candidate_checkpoint_path = candidate_checkpoint_path.expanduser().resolve()
    sft_final_path = sft_final_path.expanduser().resolve()
    policy_init_path = policy_init_path.expanduser().resolve()

    if not candidate_checkpoint_path.exists():
        raise RuntimeError(f"Candidate checkpoint does not exist: {candidate_checkpoint_path}")
    adapter_sha256 = _file_sha256(candidate_checkpoint_path)
    _archive_existing_promoted(sft_final_path, policy_init_path)

    link_or_copy_atomic(candidate_checkpoint_path, sft_final_path)
    context = CheckpointContext(
        dataset_snapshot_id=None,
        training_config_fingerprint=None,
    )
    sft_metadata = build_canonical_checkpoint_metadata(
        stage="stage1",
        run_id=run_id,
        checkpoint_role="sft_final",
        checkpoint_path=sft_final_path,
        source_checkpoint_path=candidate_checkpoint_path,
        context=context,
        metrics={
            "compile_rate": float(gate_result["candidate"]["compile_rate"]),
            "schema_rate": float(gate_result["candidate"]["schema_rate"]),
        },
        extra={
            "promoted": True,
            "adapter_sha256": adapter_sha256,
            "promoted_at": utc_now_iso8601(),
            "promotion_gate": gate_result,
        },
    )
    write_checkpoint_metadata(sft_final_path, sft_metadata)

    policy_init_action = "kept_existing"
    if force_policy_init or not policy_init_path.exists():
        link_or_copy_atomic(sft_final_path, policy_init_path)
        policy_metadata = build_canonical_checkpoint_metadata(
            stage="stage1",
            run_id=run_id,
            checkpoint_role="policy_init",
            checkpoint_path=policy_init_path,
            source_checkpoint_path=sft_final_path,
            context=context,
            metrics={
                "compile_rate": float(gate_result["candidate"]["compile_rate"]),
                "schema_rate": float(gate_result["candidate"]["schema_rate"]),
            },
            extra={
                "promoted": True,
                "adapter_sha256": adapter_sha256,
                "promoted_at": utc_now_iso8601(),
                "policy_init_immutable": not force_policy_init,
            },
        )
        write_checkpoint_metadata(policy_init_path, policy_metadata)
        policy_init_action = "updated" if force_policy_init else "created"

    return {
        "candidate_checkpoint_path": str(candidate_checkpoint_path),
        "sft_final_path": str(sft_final_path),
        "policy_init_path": str(policy_init_path),
        "policy_init_action": policy_init_action,
    }


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _archive_existing_promoted(sft_final_path: Path, policy_init_path: Path) -> None:
    archive_dir = sft_final_path.parent / "promotion_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now_iso8601().replace(":", "").replace("+", "_")
    entries: list[dict[str, Any]] = []
    for role, path in (("sft_final", sft_final_path), ("policy_init", policy_init_path)):
        if not path.exists():
            continue
        archive_path = archive_dir / f"{timestamp}_{role}_{path.name}"
        shutil.copy2(path, archive_path)
        metadata_path = path.with_name(f"{path.name}.metadata.json")
        archived_metadata_path = None
        if metadata_path.exists():
            archived_metadata_path = archive_dir / f"{timestamp}_{role}_{metadata_path.name}"
            shutil.copy2(metadata_path, archived_metadata_path)
        entries.append(
            {
                "role": role,
                "path": str(archive_path),
                "metadata_path": str(archived_metadata_path) if archived_metadata_path else None,
                "sha256": _file_sha256(archive_path),
                "archived_at": timestamp,
            }
        )
    if not entries:
        return
    manifest_path = archive_dir / "promotion_archive_manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            manifest = {}
    else:
        manifest = {}
    manifest_entries = manifest.setdefault("entries", [])
    if not isinstance(manifest_entries, list):
        manifest["entries"] = manifest_entries = []
    manifest_entries.extend(entries)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run_sft_promotion_gate(
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    baseline_key: str | None,
    candidate_key: str | None,
    min_compile_delta: float,
    min_schema_delta: float,
    min_candidate_compile_rate: float,
    min_candidate_schema_rate: float,
    baseline_compile_rate: float | None,
    candidate_compile_rate: float | None,
    promote: bool,
    candidate_checkpoint_path: Path | None,
    sft_final_path: Path,
    policy_init_path: Path,
    force_policy_init: bool,
    run_id: str,
    gate_config_path: Path | None = None,
) -> dict[str, Any]:
    gate_config = load_gate_config(gate_config_path)
    baseline_metrics = load_promotion_metrics(
        report_path=baseline_report_path,
        result_key=baseline_key,
        compile_rate_override=baseline_compile_rate,
    )
    candidate_metrics = load_promotion_metrics(
        report_path=candidate_report_path,
        result_key=candidate_key,
        compile_rate_override=candidate_compile_rate,
    )

    gate_result = evaluate_sft_promotion_gate(
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        min_compile_delta=min_compile_delta,
        min_schema_delta=min_schema_delta,
        min_candidate_compile_rate=min_candidate_compile_rate,
        min_candidate_schema_rate=min_candidate_schema_rate,
        gate_config=gate_config,
    )

    output: dict[str, Any] = {
        "evaluated_at": utc_now_iso8601(),
        "baseline_report_path": str(baseline_report_path.expanduser().resolve()),
        "candidate_report_path": str(candidate_report_path.expanduser().resolve()),
        "gate": gate_result,
        "gate_config_path": str(gate_config_path.expanduser().resolve()) if gate_config_path else None,
        "promotion": None,
    }

    if promote and gate_result["passed"]:
        if candidate_checkpoint_path is None:
            raise RuntimeError("--candidate-checkpoint is required when --promote is set and gate passes.")
        output["promotion"] = promote_sft_checkpoint(
            candidate_checkpoint_path=candidate_checkpoint_path,
            sft_final_path=sft_final_path,
            policy_init_path=policy_init_path,
            force_policy_init=force_policy_init,
            run_id=run_id,
            gate_result=gate_result,
        )

    return output
