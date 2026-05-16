#!/usr/bin/env python3
"""A/B Evaluation for TikZ Finetuning.

Compares base model vs finetuned model on a fixed set of examples.
Can also show progression across multiple checkpoints.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import random
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from tikz_mlx.compiler import CompilerService
from tikz_mlx.model_io import MlxVlmAdapter
from tikz_mlx.normalize import normalize_tikz
from tikz_mlx.schemas import CompileStatus
from tikz_mlx.prompting import extract_latex_from_response
from tikz_mlx.recovery import has_repetition_failure, substantive_features
from tikz_mlx.settings import load_config
from tikz_mlx.render_sanity import check_render_sanity
from tikz_mlx.bad_patterns import check_bad_patterns
from tikz_mlx.collapse_probe import run_collapse_probe_suite
from tikz_mlx.prompting import build_generation_prompt


# ── Adapter resolution ──────────────────────────────────────────────────────


def _resolve_adapter_dir(adapter_path: str) -> tuple[str, Any]:
    """Resolve adapter path to a directory that mlx_vlm.load() accepts."""
    p = Path(adapter_path)
    if p.is_dir():
        return str(p), None
    parent = p.parent
    config_file = parent / "adapter_config.json"
    if config_file.exists():
        if p.name == "adapters.safetensors":
            return str(parent), None
        import tempfile, shutil
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="ab_eval_adapter_")
        tmp_dir = Path(tmp_dir_obj.name)
        shutil.copy2(config_file, tmp_dir / "adapter_config.json")
        (tmp_dir / "adapters.safetensors").symlink_to(p.resolve())
        return str(tmp_dir), tmp_dir_obj
    raise RuntimeError(f"Missing adapter_config.json for checkpoint: {p}")


# ── Data loading ────────────────────────────────────────────────────────────


def _load_eval_pool(dataset_path: Path) -> list[dict]:
    """Load samples from the gold eval dataset."""
    pool: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            messages = rec.get("messages")
            if not isinstance(messages, list) or len(messages) < 2: continue
            user_text = _extract_text(messages[0])
            ref_text = _extract_text(messages[1])
            if not user_text: continue
            pool.append({
                "row_index": row_index,
                "sample_id": rec.get("sample_id", f"row_{row_index}"),
                "prompt_text": user_text,
                "reference_code": ref_text or "",
            })
    return pool


def _extract_text(msg: dict) -> str | None:
    content = msg.get("content")
    if isinstance(content, str) and content.strip(): return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip(): return text
    return None


def _select_fixed_samples(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Always select the same N samples for a given seed."""
    if len(pool) < n: n = len(pool)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pool)), n))
    return [pool[i] for i in indices]


def _select_manifest_samples(pool: list[dict], manifest_path: Path, limit: int | None = None) -> list[dict]:
    """Select samples listed by ID in the manifest."""
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    sample_ids = [str(value) for value in manifest.get("sample_ids", [])]
    pool_by_id: dict[str, dict] = {}
    for sample in pool:
        pool_by_id.setdefault(str(sample["sample_id"]), sample)
    samples = [pool_by_id[sample_id] for sample_id in sample_ids if sample_id in pool_by_id]
    if limit is not None and limit > 0:
        return samples[:limit]
    return samples


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _base_cache_key(
    *,
    cfg: Any,
    dataset_path: Path,
    manifest_path: Path | None,
    samples: list[dict],
    seed: int,
    max_tokens: int,
    prompt_contract_version: str,
    prompt_template_sha256_value: str,
) -> str:
    compiler_config = asdict(cfg.compiler) if is_dataclass(cfg.compiler) else vars(cfg.compiler)
    payload = {
        "cache_version": 2,
        "model_id": cfg.model.model_id,
        "enable_thinking": cfg.model.enable_thinking,
        "compiler_config": compiler_config,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _file_sha256(dataset_path),
        "manifest_path": str(manifest_path.resolve()) if manifest_path else None,
        "manifest_sha256": _file_sha256(manifest_path),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "seed": seed,
        "max_tokens": max_tokens,
        "prompt_contract_version": prompt_contract_version,
        "prompt_template_sha256": prompt_template_sha256_value,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ── Generation ──────────────────────────────────────────────────────────────


def _generate(
    model: Any,
    processor: Any,
    cfg: Any,
    prompt_text: str,
    max_tokens: int,
) -> str:
    """Generate a response using the loaded model."""
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=0,
        chat_template_kwargs={"enable_thinking": cfg.model.enable_thinking},
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "processor": processor,
        "prompt": prompt,
        "image": None,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "verbose": False,
    }
    sig = inspect.signature(stream_generate).parameters
    if "repetition_penalty" in sig:
        kwargs["repetition_penalty"] = 1.2

    gen = stream_generate(**kwargs)
    result = ""
    for chunk in gen:
        text_chunk = MlxVlmAdapter._coerce_generation_text(chunk)
        result += text_chunk
        if result.count("```") >= 2: break
    return result


TOKEN_FALLBACK_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def _fallback_token_count(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_FALLBACK_RE.findall(text))


def _token_count(text: str, processor: Any | None = None) -> int:
    if not text:
        return 0
    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            encoded = encode(text)
            if isinstance(encoded, list):
                return len(encoded)
            if hasattr(encoded, "__len__"):
                return len(encoded)
        except Exception:
            pass
    return _fallback_token_count(text)


def _decoding_config_to_dict(decoding: Any) -> dict[str, Any]:
    return {
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "top_k": decoding.top_k,
        "min_p": decoding.min_p,
        "repetition_penalty": decoding.repetition_penalty,
    }


def _score_generated_sample(
    generated: dict,
    compiler_config: Any,
) -> dict:
    compiler = CompilerService(compiler_config)
    raw_response = generated["raw_response"]
    latex = extract_latex_from_response(raw_response)
    wrapped_latex = normalize_tikz(latex) if latex else ""
    compile_summary = compiler.compile_document(wrapped_latex) if wrapped_latex else None
    compile_ok = (
        compile_summary is not None
        and compile_summary.status == CompileStatus.SUCCESS
    )

    has_preview = "\\PreviewEnvironment" in raw_response
    has_usepackage = "\\usepackage" in raw_response
    has_documentclass = "\\documentclass" in raw_response
    has_decorations_geometric = "decorations.geometric" in raw_response
    repetition_loop = has_repetition_failure(raw_response)
    substantive = substantive_features(latex) if latex else {"substantive_pass": False, "substantive_score": 0.0}
    true_substantive = bool(substantive.get("substantive_pass", False))
    bad_patterns = check_bad_patterns(latex) if latex else {"pass": False, "violations": ["empty_output"]}
    bad_patterns_pass = bool(bad_patterns.get("pass", False))

    render_sanity_passed = False
    if compile_ok and compile_summary.pdf_path:
        render_sanity_passed = check_render_sanity(compile_summary.pdf_path)
    closing_fence_exactly_once = raw_response.count("```") == 1

    sample = generated["sample"]
    return {
        "sample_id": sample["sample_id"],
        "prompt_text": sample["prompt_text"],
        "reference_code": sample["reference_code"],
        "raw_response": raw_response,
        "extracted_latex": latex,
        "compile_ok": compile_ok,
        "render_sanity_passed": render_sanity_passed,
        "true_substantive": true_substantive,
        "bad_patterns_pass": bad_patterns_pass,
        "bad_pattern_violations": list(bad_patterns.get("violations", [])),
        "substantive_score": float(substantive.get("substantive_score", 0.0)),
        "has_preview_env": has_preview,
        "has_usepackage": has_usepackage,
        "has_documentclass": has_documentclass,
        "has_decorations_geometric": has_decorations_geometric,
        "repetition_loop": repetition_loop,
        "closing_fence_exactly_once": closing_fence_exactly_once,
        "code_length": len(latex or ""),
        "raw_token_length": int(generated.get("raw_token_length", _fallback_token_count(raw_response))),
        "code_token_length": int(generated.get("code_token_length", _fallback_token_count(latex or ""))),
        "truncated": len(raw_response) >= generated["max_tokens"] * 3,
    }


def _evaluate_samples(
    label: str,
    adapter_path: str | None,
    samples: list[dict],
    cfg: Any,
    max_tokens: int,
    out_dir: Path,
    compile_workers: int = 1,
    collapse_probe_out: Path | None = None,
) -> list[dict]:
    """Load model/adapter and evaluate all samples."""
    print(f"\nEvaluating variant: {label}")
    adapter_dir = None
    tmp_dir_obj = None
    if adapter_path:
        adapter_dir, tmp_dir_obj = _resolve_adapter_dir(adapter_path)

    model, processor = MlxVlmAdapter.load_model(
        cfg.model.model_id,
        adapter_path=adapter_dir,
    )

    if collapse_probe_out is not None:
        probe_payload = run_collapse_probe_suite(
            model,
            processor,
            build_generation_prompt,
            verbose=True,
            production_decoding=_decoding_config_to_dict(cfg.inference.initial_decoding),
        )
        collapse_probe_out.parent.mkdir(parents=True, exist_ok=True)
        collapse_probe_out.write_text(
            json.dumps(
                {
                    **probe_payload,
                    "label": label,
                    "adapter": adapter_path,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not probe_payload["passed"]:
            raise RuntimeError(f"Collapse probe failed for {label}; details: {collapse_probe_out}")

    generated_rows = []
    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}... generated", flush=True)
        raw_response = _generate(model, processor, cfg, s["prompt_text"], max_tokens)
        latex = extract_latex_from_response(raw_response)
        generated_rows.append({
            "sample": s,
            "raw_response": raw_response,
            "max_tokens": max_tokens,
            "raw_token_length": _token_count(raw_response, processor),
            "code_token_length": _token_count(latex or "", processor),
        })

    if compile_workers > 1 and len(generated_rows) > 1:
        print(f"  Scoring/compiling with {compile_workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=compile_workers) as executor:
            results = list(executor.map(lambda row: _score_generated_sample(row, cfg.compiler), generated_rows))
    else:
        results = [_score_generated_sample(row, cfg.compiler) for row in generated_rows]
    print(f"  Scored {len(results)} samples.")
    return results


def _format_report(timestamp, config_path, dataset_path, num_samples, seed, max_tokens, samples, all_results):
    lines = ["═"*60, "TikZ A/B Evaluation Report", "═"*60, f"Generated: {timestamp}", f"Config: {config_path}", f"Dataset: {dataset_path}", f"Samples: {num_samples} (seed={seed})", f"Max tokens: {max_tokens}", ""]
    metrics = [("Compile Rate", "compile_ok"), ("Render Sanity", "render_sanity_passed"), ("Substantive", "true_substantive"), ("Bad Pattern Pass", "bad_patterns_pass"), ("Preview Env Rate", "has_preview_env"), ("Usepackage Rate", "has_usepackage"), ("Repetition Loop", "repetition_loop"), ("Fence OK Rate", "closing_fence_exactly_once")]
    header = f"{'Metric':<20}"
    for label in all_results.keys(): header += f" | {label:^12}"
    lines.append(header); lines.append("-" * len(header))
    for name, key in metrics:
        row = f"{name:<20}"
        for label, results in all_results.items():
            count = sum(1 for r in results if r[key])
            rate = count / len(results) if results else 0
            row += f" | {rate:12.1%}"
        lines.append(row)
    row = f"{'Avg Code Length':<20}"
    for label, results in all_results.items():
        avg = sum(r["code_length"] for r in results) / len(results) if results else 0
        row += f" | {avg:12.1f}"
    lines.append(row)
    row = f"{'Avg Code Tokens':<20}"
    for label, results in all_results.items():
        avg = sum(r.get("code_token_length", _fallback_token_count(r.get("extracted_latex", ""))) for r in results) / len(results) if results else 0
        row += f" | {avg:12.1f}"
    lines.append(row)
    return "\n".join(lines)


def _variant_metrics(label: str, results: list[dict], base_results: list[dict] | None = None) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {
            "compile_rate": 0.0,
            "substantive_rate": 0.0,
            "bad_pattern_pass_rate": 0.0,
            "preview_environment_rate": 0.0,
            "assistant_usepackage_rate": 0.0,
            "assistant_documentclass_rate": 0.0,
            "decorations_geometric_rate": 0.0,
            "repetition_loop_rate": 0.0,
            "closing_fence_exactly_once_rate": 0.0,
            "truncation_rate": 0.0,
            "avg_code_length": 0.0,
            "avg_code_length_ratio_vs_base": 1.0,
            "avg_raw_token_length": 0.0,
            "avg_code_token_length": 0.0,
            "avg_raw_token_ratio_vs_base": 1.0,
            "avg_code_token_ratio_vs_base": 1.0,
        }

    def avg(rows: list[dict], key: str, fallback_key: str | None = None) -> float:
        values = []
        for row in rows:
            if key in row:
                values.append(float(row[key]))
            elif fallback_key is not None:
                values.append(float(_fallback_token_count(row.get(fallback_key, ""))))
            else:
                values.append(0.0)
        return sum(values) / len(values) if values else 0.0

    base = base_results or results
    label_avg_len = avg(results, "code_length")
    base_avg_len = avg(base, "code_length")
    label_raw_tokens = avg(results, "raw_token_length", "raw_response")
    base_raw_tokens = avg(base, "raw_token_length", "raw_response")
    label_code_tokens = avg(results, "code_token_length", "extracted_latex")
    base_code_tokens = avg(base, "code_token_length", "extracted_latex")
    return {
        "compile_rate": sum(1 for r in results if r["compile_ok"]) / n,
        "substantive_rate": sum(1 for r in results if r["true_substantive"]) / n,
        "bad_pattern_pass_rate": sum(1 for r in results if r["bad_patterns_pass"]) / n,
        "preview_environment_rate": sum(1 for r in results if r["has_preview_env"]) / n,
        "assistant_usepackage_rate": sum(1 for r in results if r["has_usepackage"]) / n,
        "assistant_documentclass_rate": sum(1 for r in results if r["has_documentclass"]) / n,
        "decorations_geometric_rate": sum(1 for r in results if r["has_decorations_geometric"]) / n,
        "repetition_loop_rate": sum(1 for r in results if r["repetition_loop"]) / n,
        "closing_fence_exactly_once_rate": sum(1 for r in results if r["closing_fence_exactly_once"]) / n,
        "truncation_rate": sum(1 for r in results if r.get("truncated")) / n,
        "avg_code_length": label_avg_len,
        "avg_code_length_ratio_vs_base": label_avg_len / base_avg_len if base_avg_len > 0 else 1.0,
        "avg_raw_token_length": label_raw_tokens,
        "avg_code_token_length": label_code_tokens,
        "avg_raw_token_ratio_vs_base": label_raw_tokens / base_raw_tokens if base_raw_tokens > 0 else 1.0,
        "avg_code_token_ratio_vs_base": label_code_tokens / base_code_tokens if base_code_tokens > 0 else 1.0,
    }


def _load_or_compute_base_results(
    *,
    cfg: Any,
    config_path: str,
    dataset_path: Path,
    manifest_path: Path | None,
    samples: list[dict],
    seed: int,
    max_tokens: int,
    output_root: Path,
    base_cache_dir: Path,
    prompt_contract_version: str,
    prompt_template_sha256_value: str,
    compile_workers: int,
) -> list[dict]:
    cache_key = _base_cache_key(
        cfg=cfg,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        samples=samples,
        seed=seed,
        max_tokens=max_tokens,
        prompt_contract_version=prompt_contract_version,
        prompt_template_sha256_value=prompt_template_sha256_value,
    )
    cache_path = base_cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        print(f"Loaded cached base eval: {cache_path}")
        return list(payload["results"])

    results = _evaluate_samples(
        "base",
        None,
        samples,
        cfg,
        max_tokens,
        output_root,
        compile_workers=compile_workers,
    )
    base_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "config_path": config_path,
                "dataset_path": str(dataset_path),
                "manifest_path": str(manifest_path) if manifest_path else None,
                "sample_ids": [sample["sample_id"] for sample in samples],
                "max_tokens": max_tokens,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote cached base eval: {cache_path}")
    return results


def run_ab_eval(config_path, dataset_path=None, eval_manifest_path=None, sentinel_manifest_path=None, adapter_path=None, checkpoint_dir=None, num_samples=120, seed=42, max_tokens=4096, skip_base=False, skip_finetuned=False, out_dir=None, base_cache_dir=None, no_base_cache=False, collapse_probe_out=None, skip_collapse_probe=False, compile_workers=1):
    from tikz_mlx.prompting import PROMPT_CONTRACT_VERSION, prompt_template_sha256
    cfg = load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_path = Path(dataset_path or eval_manifest_path or cfg.training.gold_eval_dataset_path)
    manifest_path = Path(sentinel_manifest_path) if sentinel_manifest_path else None
    pool = _load_eval_pool(ds_path)
    if manifest_path:
        samples = _select_manifest_samples(pool, manifest_path, limit=num_samples)
    else:
        samples = _select_fixed_samples(pool, num_samples, seed)
    if not samples:
        raise RuntimeError(
            "A/B eval selected zero samples. Check the eval dataset, sentinel manifest, "
            f"and --num-samples value (dataset={ds_path}, manifest={manifest_path})."
        )
    output_root = Path(out_dir or (Path(cfg.paths.outputs_dir) / f"ab_eval_{timestamp}"))
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = {}
    prompt_sha = prompt_template_sha256()
    if not skip_base:
        if base_cache_dir and not no_base_cache:
            all_results["base"] = _load_or_compute_base_results(
                cfg=cfg,
                config_path=config_path,
                dataset_path=ds_path,
                manifest_path=manifest_path,
                samples=samples,
                seed=seed,
                max_tokens=max_tokens,
                output_root=output_root,
                base_cache_dir=Path(base_cache_dir),
                prompt_contract_version=PROMPT_CONTRACT_VERSION,
                prompt_template_sha256_value=prompt_sha,
                compile_workers=compile_workers,
            )
        else:
            all_results["base"] = _evaluate_samples(
                "base",
                None,
                samples,
                cfg,
                max_tokens,
                output_root,
                compile_workers=compile_workers,
            )
    if adapter_path and not skip_finetuned:
        all_results["finetuned"] = _evaluate_samples(
            "finetuned",
            adapter_path,
            samples,
            cfg,
            max_tokens,
            output_root,
            compile_workers=compile_workers,
            collapse_probe_out=Path(collapse_probe_out) if collapse_probe_out and not skip_collapse_probe else None,
        )
    report = _format_report(timestamp, config_path, str(ds_path), len(samples), seed, max_tokens, samples, all_results)
    (output_root / "report.txt").write_text(report, encoding="utf-8")
    structured = {"timestamp": timestamp, "config_path": config_path, "dataset_path": str(ds_path), "num_samples": len(samples), "seed": seed, "max_tokens": max_tokens, "sample_ids": [s["sample_id"] for s in samples], "prompt_contract_version": PROMPT_CONTRACT_VERSION, "prompt_template_sha256": prompt_sha}
    worst_examples = []
    for label, results in all_results.items():
        n = len(results)
        label_avg_len = sum(r["code_length"] for r in results) / n if n else 0
        base_avg_len = sum(r["code_length"] for r in all_results.get("base", [])) / len(all_results.get("base", [])) if all_results.get("base") else label_avg_len
        for r in results:
            is_worst = (not r["compile_ok"] or r["has_preview_env"] or r["has_usepackage"] or r["repetition_loop"] or (r["code_length"] > 1.8 * base_avg_len and r["code_length"] > 500))
            if is_worst and label != "base":
                worst_examples.append({"label": label, "sample_id": r["sample_id"], "prompt": r["prompt_text"], "reference": r["reference_code"], "response": r["raw_response"], "failures": [k for k, v in r.items() if k in ["has_preview_env", "has_usepackage", "repetition_loop", "compile_ok"] and (v if k != "compile_ok" else not v)]})
        structured[label] = _variant_metrics(label, results, all_results.get("base"))
    (output_root / "results.json").write_text(json.dumps(structured, indent=2), encoding="utf-8")
    if worst_examples: (output_root / "worst_cases.json").write_text(json.dumps(worst_examples[:20], indent=2), encoding="utf-8")
    print("\n" + report, flush=True); return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="TikZ A/B Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--sentinel-manifest", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-finetuned", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--base-cache-dir", default=None)
    parser.add_argument("--no-base-cache", action="store_true")
    parser.add_argument("--collapse-probe-out", default=None)
    parser.add_argument("--skip-collapse-probe", action="store_true")
    parser.add_argument("--compile-workers", type=int, default=1)
    args = parser.parse_args()
    run_ab_eval(
        args.config,
        args.dataset,
        args.eval_manifest,
        args.sentinel_manifest,
        args.adapter_path,
        args.checkpoint_dir,
        args.num_samples,
        args.seed,
        args.max_tokens,
        args.skip_base,
        args.skip_finetuned,
        args.out_dir,
        args.base_cache_dir,
        args.no_base_cache,
        args.collapse_probe_out,
        args.skip_collapse_probe,
        args.compile_workers,
    )

if __name__ == "__main__":
    main()
