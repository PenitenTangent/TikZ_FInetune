#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tikz_mlx.compiler import CompilerService
from tikz_mlx.dataset import iter_jsonl
from tikz_mlx.debug_render import load_render_config, prepare_image_for_reward_encoder, rasterize_pdf
from tikz_mlx.model_io import LoadedModel, MlxVlmAdapter
from tikz_mlx.normalize import normalize_tikz
from tikz_mlx.prompting import extract_latex_from_response
from tikz_mlx.recovery import has_repetition_failure
from tikz_mlx.reward.emd import EarthMoverReward
from tikz_mlx.reward.encoder_detikzify import FrozenDetikzifyEncoder
from tikz_mlx.reward.selfsim import SelfSimReward
from tikz_mlx.schema_validation import is_canonical_tikz_document
from tikz_mlx.settings import load_config


_TOKEN_RE = re.compile(r"\\[A-Za-z@]+|\d+(?:\.\d+)?|[A-Za-z]+|[^\sA-Za-z0-9]")
_COMMAND_RE = re.compile(r"\\[A-Za-z@]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_DOCUMENT_BEGIN_RE = re.compile(r"\\begin\{document\}", re.IGNORECASE)
_DOCUMENT_END_RE = re.compile(r"\\end\{document\}", re.IGNORECASE)


def _strip_tex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        lines.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(lines)


def _strip_wrappers(text: str) -> str:
    cleaned = text.strip()
    begin_match = _DOCUMENT_BEGIN_RE.search(cleaned)
    end_match = _DOCUMENT_END_RE.search(cleaned)
    if begin_match and end_match and begin_match.start() < end_match.start():
        return cleaned[begin_match.end():end_match.start()].strip()

    env_match = re.search(r"\\begin\{(?:tikzpicture|tikz-cd|circuitikz|axis)\}", cleaned, flags=re.IGNORECASE)
    if env_match:
        return cleaned[env_match.start():].strip()

    return cleaned


def _tokenize_tex(text: str) -> set[str]:
    cleaned = _strip_tex_comments(text)
    return {token for token in _TOKEN_RE.findall(cleaned) if token.strip()}


def _token_jaccard(ref: str, code: str) -> float:
    ref_tokens = _tokenize_tex(ref)
    code_tokens = _tokenize_tex(code)
    if not ref_tokens and not code_tokens:
        return 1.0
    if not ref_tokens or not code_tokens:
        return 0.0
    return len(ref_tokens & code_tokens) / len(ref_tokens | code_tokens)


def _command_number_mix(ref: str, code: str) -> float:
    ref_commands = set(_COMMAND_RE.findall(_strip_tex_comments(ref)))
    code_commands = set(_COMMAND_RE.findall(_strip_tex_comments(code)))
    ref_numbers = set(_NUMBER_RE.findall(_strip_tex_comments(ref)))
    code_numbers = set(_NUMBER_RE.findall(_strip_tex_comments(code)))

    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    return 0.7 * _jaccard(ref_commands, code_commands) + 0.3 * _jaccard(ref_numbers, code_numbers)


def _score_with_fallback(
    ref_emb: Any,
    cand_emb: Any,
    emd_backend: Any,
    selfsim_backend: Any,
) -> dict[str, Any]:
    try:
        score = float(emd_backend.score_embeddings(ref_emb, cand_emb))
        return {
            "backend_used": "emd",
            "fallback_used": False,
            "fallback_reason": None,
            "score": max(0.0, min(1.0, score)),
        }
    except Exception as exc:
        score = float(selfsim_backend.score_embeddings(ref_emb, cand_emb))
        return {
            "backend_used": "selfsim",
            "fallback_used": True,
            "fallback_reason": str(exc),
            "score": max(0.0, min(1.0, score)),
        }


def _extract_prompt_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("Dataset record missing messages.")

    first = messages[0]
    if not isinstance(first, dict):
        raise RuntimeError("Dataset record contains a malformed user message.")
    content = first.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for piece in content:
            if isinstance(piece, dict):
                text = piece.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    raise RuntimeError("Dataset record does not contain a usable prompt text.")


def _extract_reference_code(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise RuntimeError("Dataset record missing assistant reference message.")

    assistant = messages[1]
    if not isinstance(assistant, dict):
        raise RuntimeError("Dataset record contains a malformed assistant message.")
    content = assistant.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for piece in content:
            if isinstance(piece, dict):
                text = piece.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    raise RuntimeError("Dataset record does not contain a usable reference code block.")


def _canonical_document_from_prompt(prompt_text: str, completion_text: str) -> str:
    prompt_prefix = prompt_text.split("```latex", 1)[0].rstrip()
    completion = extract_latex_from_response(completion_text)
    completion = _strip_wrappers(completion)
    return f"{prompt_prefix}\n{completion}".strip() + "\n"


def _build_reference_document(record: dict[str, Any]) -> str:
    prompt_text = _extract_prompt_text(record)
    reference_text = _extract_reference_code(record)
    return _canonical_document_from_prompt(prompt_text, reference_text)


def _build_candidate_document(record: dict[str, Any], response_text: str) -> str:
    prompt_text = _extract_prompt_text(record)
    return _canonical_document_from_prompt(prompt_text, response_text)


def _select_samples(records: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size >= len(records):
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_size))
    return [records[index] for index in indices]


def _safe_bool(value: Any) -> bool:
    return bool(value)


class VariantTotals:
    def __init__(self) -> None:
        self.total = 0
        self.compile_success = 0
        self.schema_success = 0
        self.hybrid_pass = 0
        self.repetition_failure = 0
        self.truncation = 0
        self.substantive_tikz = 0
        self.substantive_compile = 0
        self.score_sum = 0.0
        self.emd_sum = 0.0
        self.emd_count = 0

    def to_summary(self, per_prompt: list[dict[str, Any]]) -> dict[str, Any]:
        total = self.total or len(per_prompt)
        compile_rate = self.compile_success / total if total else 0.0
        schema_rate = self.schema_success / total if total else 0.0
        hybrid_pass_rate = self.hybrid_pass / total if total else 0.0
        repetition_loop_rate = self.repetition_failure / total if total else 0.0
        truncation_rate = self.truncation / total if total else 0.0
        substantive_tikz_rate = self.substantive_tikz / total if total else 0.0
        substantive_compile_success_rate = self.substantive_compile / total if total else 0.0
        mean_hybrid_score = self.score_sum / total if total else 0.0
        mean_emd = self.emd_sum / self.emd_count if self.emd_count else None
        return {
            "total": total,
            "compile_rate": compile_rate,
            "substantive_compile_success_rate": substantive_compile_success_rate,
            "substantive_tikz_rate": substantive_tikz_rate,
            "schema_rate": schema_rate,
            "schema_rate_source": "schema_rate",
            "hybrid_pass_rate": hybrid_pass_rate,
            "repetition_loop_rate": repetition_loop_rate,
            "truncation_rate": truncation_rate,
            "mean_hybrid_score": mean_hybrid_score,
            "mean_emd": mean_emd,
            "per_prompt": per_prompt,
        }


class HybridScorer:
    def __init__(
        self,
        *,
        cfg: Any,
        compiler: CompilerService | None,
        out_root: Path,
        similarity_mode: str,
        prefilter_threshold: float,
        visual_score_threshold: float,
        reward_backend: str,
        reward_model_id: str,
        reference_code_by_row: dict[int, str],
        render_config: Any,
        phase_a_token_weight: float,
        phase_a_command_weight: float,
        phase_b_blend_weight: float,
        hybrid_combine_mode: str,
        hybrid_score_gamma: float,
    ) -> None:
        self.cfg = cfg
        self.compiler = compiler
        self.out_root = out_root
        self.similarity_mode = similarity_mode
        self.prefilter_threshold = prefilter_threshold
        self.visual_score_threshold = visual_score_threshold
        self.reward_backend = reward_backend
        self.reward_model_id = reward_model_id
        self.reference_code_by_row = reference_code_by_row
        self.render_config = render_config
        self.phase_a_token_weight = phase_a_token_weight
        self.phase_a_command_weight = phase_a_command_weight
        self.phase_b_blend_weight = phase_b_blend_weight
        self.hybrid_combine_mode = hybrid_combine_mode
        self.hybrid_score_gamma = hybrid_score_gamma
        self._reference_embedding_cache: dict[int, Any] = {}
        self._encoder: FrozenDetikzifyEncoder | None = None
        self._emd_backend = EarthMoverReward()
        self._selfsim_backend = SelfSimReward()

    def _get_encoder(self) -> FrozenDetikzifyEncoder:
        if self._encoder is None:
            self._encoder = FrozenDetikzifyEncoder(self.reward_model_id, self.cfg.memory)
        return self._encoder

    def _get_reference_embedding(self, row_index: int) -> Any:
        if row_index in self._reference_embedding_cache:
            return self._reference_embedding_cache[row_index]

        reference_code = self.reference_code_by_row[row_index]
        reference_dir = self.out_root / "reference" / f"row_{row_index:05d}"
        reference_dir.mkdir(parents=True, exist_ok=True)
        compile_summary = self.compiler.compile_document(reference_code, output_dir=reference_dir, job_name="reference") if self.compiler else None
        if compile_summary is None or compile_summary.pdf_path is None:
            self._reference_embedding_cache[row_index] = []
            return []

        reference_png = rasterize_pdf(compile_summary.pdf_path, reference_dir, render_config=self.render_config)
        prepared = prepare_image_for_reward_encoder(reference_png, self.render_config)
        embedding = self._get_encoder().encode_image(str(prepared))
        self._reference_embedding_cache[row_index] = embedding
        return embedding

    def score(
        self,
        *,
        sample: dict[str, Any],
        normalized_code: str,
        substantive_compile_success: bool,
        sample_dir: Path,
        candidate_pdf_path: str | None,
    ) -> dict[str, Any]:
        row_index = int(sample["row_index"])
        reference_code = self.reference_code_by_row[row_index]

        phase_a_token_jaccard = _token_jaccard(reference_code, normalized_code)
        phase_a_command_number_mix = _command_number_mix(reference_code, normalized_code)
        phase_a_score = (
            self.phase_a_token_weight * phase_a_token_jaccard
            + self.phase_a_command_weight * phase_a_command_number_mix
        ) / max(self.phase_a_token_weight + self.phase_a_command_weight, 1e-12)
        code_prefilter_pass = phase_a_score >= self.prefilter_threshold

        phase_b_score: float | None = None
        hybrid_pass_reason: str | None = None
        hybrid_pass = False
        backend_used: str | None = None
        fallback_used = False
        fallback_reason: str | None = None
        emd_score: float | None = None

        if substantive_compile_success and candidate_pdf_path is not None:
            candidate_png = rasterize_pdf(candidate_pdf_path, sample_dir, render_config=self.render_config)
            prepared_candidate = prepare_image_for_reward_encoder(candidate_png, self.render_config)
            candidate_embedding = self._get_encoder().encode_image(str(prepared_candidate))
            reference_embedding = self._get_reference_embedding(row_index)
            reward_result = _score_with_fallback(
                reference_embedding,
                candidate_embedding,
                self._emd_backend,
                self._selfsim_backend,
            )
            backend_used = reward_result["backend_used"]
            fallback_used = _safe_bool(reward_result["fallback_used"])
            fallback_reason = reward_result["fallback_reason"]
            phase_b_score = float(reward_result["score"])
            emd_score = phase_b_score if backend_used == "emd" else None
            if not code_prefilter_pass:
                hybrid_pass_reason = "phase_b_only"
                hybrid_pass = phase_b_score >= self.visual_score_threshold
            else:
                hybrid_pass_reason = "phase_b_and_prefilter"
                hybrid_pass = phase_b_score >= self.visual_score_threshold and phase_a_score >= self.prefilter_threshold

        if phase_b_score is None:
            hybrid_score = phase_a_score
        else:
            hybrid_score = phase_b_score ** max(self.hybrid_score_gamma, 1e-12)

        if phase_b_score is not None and not code_prefilter_pass and phase_b_score >= self.visual_score_threshold:
            hybrid_pass_reason = "phase_b_only"
            hybrid_pass = True

        schema_pass = is_canonical_tikz_document(normalized_code)
        repetition_loop = has_repetition_failure(normalized_code) or has_repetition_failure(sample.get("raw_response", ""))
        truncated = "\\end{document}" not in sample.get("raw_response", "") and "\\end{document}" not in normalized_code

        return {
            "sample_id": sample.get("sample_id"),
            "row_index": row_index,
            "prompt_text": sample.get("prompt_text"),
            "reference_code": reference_code,
            "raw_response": sample.get("raw_response"),
            "normalized_code": normalized_code,
            "tex_path": str(sample_dir / "candidate.tex"),
            "log_path": str(sample_dir / "candidate.log"),
            "working_dir": str(sample_dir),
            "candidate_pdf_path": candidate_pdf_path,
            "substantive_compile_success": substantive_compile_success,
            "compile_ok": substantive_compile_success,
            "schema_pass": schema_pass,
            "truncated": truncated,
            "repetition_loop": repetition_loop,
            "phase_a_token_jaccard": phase_a_token_jaccard,
            "phase_a_command_number_mix": phase_a_command_number_mix,
            "phase_a_score": phase_a_score,
            "code_prefilter_pass": code_prefilter_pass,
            "phase_b_score": phase_b_score,
            "emd_score": emd_score,
            "hybrid_score": hybrid_score,
            "hybrid_pass": hybrid_pass,
            "hybrid_pass_reason": hybrid_pass_reason,
            "reward_backend_used": backend_used,
            "reward_fallback_used": fallback_used,
            "reward_fallback_reason": fallback_reason,
            "substantive_tikz_pass": schema_pass,
        }


def _load_records(dataset_path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(dataset_path))


def _generate_variant_samples(
    *,
    label: str,
    adapter_path: str | None,
    samples: list[dict[str, Any]],
    cfg: Any,
    compiler: CompilerService,
    out_dir: Path,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], VariantTotals]:
    adapter = MlxVlmAdapter(cfg.model, cfg.memory)
    scorer = None
    reference_code_by_row = {int(sample["row_index"]): sample["reference_code"] for sample in samples}
    render_config = load_render_config(cfg.paths.root_dir)
    scorer = HybridScorer(
        cfg=cfg,
        compiler=compiler,
        out_root=out_dir,
        similarity_mode="hybrid",
        prefilter_threshold=0.80,
        visual_score_threshold=0.75,
        reward_backend="emd",
        reward_model_id=cfg.training.stage2.reward_model_id,
        reference_code_by_row=reference_code_by_row,
        render_config=render_config,
        phase_a_token_weight=0.75,
        phase_a_command_weight=0.25,
        phase_b_blend_weight=0.70,
        hybrid_combine_mode="phase_b_only",
        hybrid_score_gamma=1.0,
    )

    per_prompt: list[dict[str, Any]] = []
    totals = VariantTotals()

    for index, sample in enumerate(samples):
        sample_dir = out_dir / label / f"sample_{index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = sample["prompt_text"]
        from mlx_vlm import load

        model, processor = load(
            cfg.model.model_id,
            adapter_path=str(adapter_path) if adapter_path else None,
            processor_config={"trust_remote_code": True},
        )
        adapter.loaded = LoadedModel(model=model, processor=processor)
        generation = adapter.generate(
            GenerationRequest(
                description=prompt_text,
                image_paths=[],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=64,
                min_p=0.05,
                repetition_penalty=1.1,
            )
        )
        raw_response = generation.text
        normalized_code = _build_candidate_document(sample, raw_response)
        candidate_tex_path = sample_dir / "candidate.tex"
        candidate_tex_path.write_text(normalized_code, encoding="utf-8")

        compile_summary = compiler.compile_document(normalized_code, output_dir=sample_dir, job_name="candidate")
        substantive_compile_success = bool(compile_summary.pdf_path and Path(compile_summary.pdf_path).exists())

        score_record = scorer.score(
            sample={**sample, "raw_response": raw_response},
            normalized_code=normalized_code,
            substantive_compile_success=substantive_compile_success,
            sample_dir=sample_dir,
            candidate_pdf_path=str(compile_summary.pdf_path) if compile_summary.pdf_path else None,
        )
        score_record.update(
            {
                "compile_status": compile_summary.status.value,
                "return_code": compile_summary.return_code,
                "key_errors": compile_summary.key_errors,
                "line_hints": compile_summary.line_hints,
                "missing_packages": compile_summary.missing_packages,
                "tex_path": str(candidate_tex_path),
                "log_path": str(compile_summary.log_path) if compile_summary.log_path else str(sample_dir / "candidate.log"),
                "working_dir": str(sample_dir),
            }
        )
        per_prompt.append(score_record)

        totals.total += 1
        totals.compile_success += int(substantive_compile_success)
        totals.schema_success += int(score_record["schema_pass"])
        totals.hybrid_pass += int(score_record["hybrid_pass"])
        totals.repetition_failure += int(score_record["repetition_loop"])
        totals.truncation += int(score_record["truncated"])
        totals.substantive_tikz += int(score_record["substantive_tikz_pass"])
        totals.substantive_compile += int(substantive_compile_success)
        totals.score_sum += float(score_record["hybrid_score"])
        if score_record["emd_score"] is not None:
            totals.emd_sum += float(score_record["emd_score"])
            totals.emd_count += 1

    return per_prompt, totals


def _build_results_block(per_prompt: list[dict[str, Any]], totals: VariantTotals) -> dict[str, Any]:
    return totals.to_summary(per_prompt)


def run_strict_ab_eval(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    adapter_dir: str | Path,
    sample_size: int,
    seed: int,
    max_tokens_cap: int,
    hybrid_visual_threshold: float,
    out_dir: str | Path,
    reward_backend: str,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    dataset_path = Path(dataset_path)
    adapter_dir = Path(adapter_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(dataset_path)
    selected = _select_samples(records, sample_size, seed)
    samples: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        prompt_text = _extract_prompt_text(record)
        samples.append(
            {
                "sample_id": record.get("sample_id", f"row_{index}"),
                "row_index": index,
                "prompt_text": prompt_text,
                "reference_code": _build_reference_document(record),
                "source_record": record,
            }
        )

    compiler = CompilerService(cfg.compiler)
    render_config = load_render_config(cfg.paths.root_dir)
    _ = render_config

    def _evaluate_block(label: str, variant_adapter: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        variant_out = out_dir / label
        variant_out.mkdir(parents=True, exist_ok=True)
        adapter = MlxVlmAdapter(cfg.model, cfg.memory)
        # Preserve the adapter path contract even though MlxVlmAdapter lazily resolves it.
        if variant_adapter is not None:
            _ = variant_adapter
        per_prompt: list[dict[str, Any]] = []
        totals = VariantTotals()
        scorer = HybridScorer(
            cfg=cfg,
            compiler=compiler,
            out_root=variant_out,
            similarity_mode="hybrid",
            prefilter_threshold=0.80,
            visual_score_threshold=hybrid_visual_threshold,
            reward_backend=reward_backend,
            reward_model_id=cfg.training.stage2.reward_model_id,
            reference_code_by_row={sample["row_index"]: sample["reference_code"] for sample in samples},
            render_config=load_render_config(cfg.paths.root_dir),
            phase_a_token_weight=0.75,
            phase_a_command_weight=0.25,
            phase_b_blend_weight=0.70,
            hybrid_combine_mode="phase_b_only",
            hybrid_score_gamma=1.0,
        )
        for index, sample in enumerate(samples):
            sample_dir = variant_out / f"sample_{index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            from mlx_vlm import load

            model, processor = load(
                cfg.model.model_id,
                adapter_path=str(variant_adapter) if variant_adapter else None,
                processor_config={"trust_remote_code": True},
            )
            adapter.loaded = LoadedModel(model=model, processor=processor)
            generation = adapter.generate(
                GenerationRequest(
                    description=sample["prompt_text"],
                    image_paths=[],
                    max_tokens=max_tokens_cap,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=64,
                    min_p=0.05,
                    repetition_penalty=1.1,
                )
            )
            raw_response = generation.text
            normalized_code = _build_candidate_document(sample, raw_response)
            candidate_tex_path = sample_dir / "candidate.tex"
            candidate_tex_path.write_text(normalized_code, encoding="utf-8")
            compile_summary = compiler.compile_document(normalized_code, output_dir=sample_dir, job_name="candidate")
            substantive_compile_success = bool(compile_summary.pdf_path and Path(compile_summary.pdf_path).exists())

            score_record = scorer.score(
                sample={**sample, "raw_response": raw_response},
                normalized_code=normalized_code,
                substantive_compile_success=substantive_compile_success,
                sample_dir=sample_dir,
                candidate_pdf_path=str(compile_summary.pdf_path) if compile_summary.pdf_path else None,
            )
            score_record.update(
                {
                    "compile_status": compile_summary.status.value,
                    "return_code": compile_summary.return_code,
                    "key_errors": compile_summary.key_errors,
                    "line_hints": compile_summary.line_hints,
                    "missing_packages": compile_summary.missing_packages,
                    "tex_path": str(candidate_tex_path),
                    "log_path": str(compile_summary.log_path) if compile_summary.log_path else str(sample_dir / "candidate.log"),
                    "working_dir": str(sample_dir),
                }
            )
            per_prompt.append(score_record)

            totals.total += 1
            totals.compile_success += int(substantive_compile_success)
            totals.schema_success += int(score_record["schema_pass"])
            totals.hybrid_pass += int(score_record["hybrid_pass"])
            totals.repetition_failure += int(score_record["repetition_loop"])
            totals.truncation += int(score_record["truncated"])
            totals.substantive_tikz += int(score_record["substantive_tikz_pass"])
            totals.substantive_compile += int(substantive_compile_success)
            totals.score_sum += float(score_record["hybrid_score"])
            if score_record["emd_score"] is not None:
                totals.emd_sum += float(score_record["emd_score"])
                totals.emd_count += 1

        return per_prompt, _build_results_block(per_prompt, totals)

    base_per_prompt, base_summary = _evaluate_block("base", None)
    stage1_per_prompt, stage1_summary = _evaluate_block("stage1", str(adapter_dir))

    payload = {
        "config_path": str(Path(config_path).expanduser().resolve()),
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "adapter_dir": str(adapter_dir.expanduser().resolve()),
        "sample_size": sample_size,
        "seed": seed,
        "max_tokens_cap": max_tokens_cap,
        "hybrid_visual_threshold": hybrid_visual_threshold,
        "reward_backend": reward_backend,
        "results": {
            "base": base_summary,
            "stage1": stage1_summary,
        },
        "per_prompt": {
            "base": base_per_prompt,
            "stage1": stage1_per_prompt,
        },
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report_txt = out_dir / "report.txt"
    report_txt.write_text(
        "\n".join(
            [
                f"Strict Stage1 A/B Evaluation",
                f"dataset={payload['dataset_path']}",
                f"sample_size={sample_size}",
                f"seed={seed}",
                f"base_compile_rate={base_summary['compile_rate']:.3f}",
                f"stage1_compile_rate={stage1_summary['compile_rate']:.3f}",
                f"base_schema_rate={base_summary['schema_rate']:.3f}",
                f"stage1_schema_rate={stage1_summary['schema_rate']:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict Stage1 A/B evaluator for TikZ finetuning.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens-cap", type=int, default=2048)
    parser.add_argument("--hybrid-visual-threshold", type=float, default=0.75)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reward-backend", default="emd")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_strict_ab_eval(
        config_path=args.config,
        dataset_path=args.dataset,
        adapter_dir=args.adapter_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        max_tokens_cap=args.max_tokens_cap,
        hybrid_visual_threshold=args.hybrid_visual_threshold,
        out_dir=args.out_dir,
        reward_backend=args.reward_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())