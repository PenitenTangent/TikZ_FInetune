#!/usr/bin/env python3
"""A/B Evaluation for TikZ Finetuning.

Compares base model vs finetuned model on a fixed set of examples.
Can also show progression across multiple checkpoints.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from tikz_mlx.compiler import CompilerService
from tikz_mlx.mlx_runtime import import_mlx_core
from tikz_mlx.model_io import MlxVlmAdapter
from tikz_mlx.normalize import normalize_tikz
from tikz_mlx.prompting import extract_latex_from_response
from tikz_mlx.recovery import has_repetition_failure, substantive_features
from tikz_mlx.settings import load_config
from tikz_mlx.render_sanity import check_render_sanity
from tikz_mlx.bad_patterns import check_bad_patterns
from tikz_mlx.token_stats import token_distribution_features
from tikz_mlx.normalization_audit import normalize_with_audit

ENV_RE = re.compile(
    r"\\begin\{(?:tikzpicture|tikz-cd|circuitikz|axis)\}", re.IGNORECASE
)
CMD_RE = re.compile(
    r"\\(?:draw|node|path|fill|filldraw|addplot|arrow)\b", re.IGNORECASE
)


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


def _select_manifest_samples(pool: list[dict], manifest_path: Path) -> list[dict]:
    """Select samples listed by ID in the manifest."""
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    ids = set(manifest.get("sample_ids", []))
    return [s for s in pool if s["sample_id"] in ids]


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
        kwargs["repetition_penalty"] = 1.15

    gen = stream_generate(**kwargs)
    result = ""
    for chunk in gen:
        text_chunk = MlxVlmAdapter._coerce_generation_text(chunk)
        result += text_chunk
        if result.count("```") >= 2: break
    return result


def _evaluate_samples(
    label: str,
    adapter_path: str | None,
    samples: list[dict],
    cfg: Any,
    compiler: CompilerService,
    max_tokens: int,
    out_dir: Path,
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

    results = []
    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}...", end="", flush=True)
        raw_response = _generate(model, processor, cfg, s["prompt_text"], max_tokens)
        latex = extract_latex_from_response(raw_response)
        compile_summary = compiler.compile(latex) if latex else None
        
        has_preview = "\\PreviewEnvironment" in raw_response
        has_usepackage = "\\usepackage" in raw_response
        has_documentclass = "\\documentclass" in raw_response
        has_decorations_geometric = "decorations.geometric" in raw_response
        repetition_loop = has_repetition_failure(raw_response)
        true_substantive = substantive_features(latex) if latex else False
        bad_patterns_pass = check_bad_patterns(latex) if latex else True
        
        render_sanity_passed = False
        if compile_summary and compile_summary.success:
            render_sanity_passed = check_render_sanity(compile_summary.pdf_path)
        closing_fence_exactly_once = raw_response.count("```") == 1

        results.append({
            "sample_id": s["sample_id"],
            "prompt_text": s["prompt_text"],
            "reference_code": s["reference_code"],
            "raw_response": raw_response,
            "extracted_latex": latex,
            "compile_ok": compile_summary.success if compile_summary else False,
            "render_sanity_passed": render_sanity_passed,
            "true_substantive": true_substantive,
            "bad_patterns_pass": bad_patterns_pass,
            "has_preview_env": has_preview,
            "has_usepackage": has_usepackage,
            "has_documentclass": has_documentclass,
            "has_decorations_geometric": has_decorations_geometric,
            "repetition_loop": repetition_loop,
            "closing_fence_exactly_once": closing_fence_exactly_once,
            "code_length": len(latex or ""),
            "truncated": len(raw_response) >= max_tokens * 3,
        })
        print(" done.")
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
    return "\n".join(lines)


def run_ab_eval(config_path, dataset_path=None, eval_manifest_path=None, sentinel_manifest_path=None, adapter_path=None, checkpoint_dir=None, num_samples=120, seed=42, max_tokens=4096, skip_base=False, skip_finetuned=False, out_dir=None):
    from tikz_mlx.prompting import PROMPT_CONTRACT_VERSION, prompt_template_sha256
    cfg = load_config(config_path)
    compiler = CompilerService(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_path = Path(dataset_path or eval_manifest_path or cfg.training.gold_eval_dataset_path)
    pool = _load_eval_pool(ds_path)
    if sentinel_manifest_path:
        samples = _select_manifest_samples(pool, Path(sentinel_manifest_path))
    else:
        samples = _select_fixed_samples(pool, num_samples, seed)
    output_root = Path(out_dir or (Path(cfg.paths.outputs_dir) / f"ab_eval_{timestamp}"))
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = {}
    adapters = []
    if not skip_base: adapters.append(("base", None))
    if adapter_path and not skip_finetuned: adapters.append(("finetuned", adapter_path))
    for label, a_path in adapters:
        all_results[label] = _evaluate_samples(label, a_path, samples, cfg, compiler, max_tokens, output_root)
    report = _format_report(timestamp, config_path, str(ds_path), len(samples), seed, max_tokens, samples, all_results)
    (output_root / "report.txt").write_text(report, encoding="utf-8")
    structured = {"timestamp": timestamp, "config_path": config_path, "dataset_path": str(ds_path), "num_samples": len(samples), "seed": seed, "max_tokens": max_tokens, "sample_ids": [s["sample_id"] for s in samples], "prompt_contract_version": PROMPT_CONTRACT_VERSION, "prompt_template_sha256": prompt_template_sha256()}
    worst_examples = []
    for label, results in all_results.items():
        n = len(results)
        label_avg_len = sum(r["code_length"] for r in results) / n if n else 0
        base_avg_len = sum(r["code_length"] for r in all_results.get("base", [])) / len(all_results.get("base", [])) if all_results.get("base") else label_avg_len
        for r in results:
            is_worst = (not r["compile_ok"] or r["has_preview_env"] or r["has_usepackage"] or r["repetition_loop"] or (r["code_length"] > 1.8 * base_avg_len and r["code_length"] > 500))
            if is_worst and label != "base":
                worst_examples.append({"label": label, "sample_id": r["sample_id"], "prompt": r["prompt_text"], "reference": r["reference_code"], "response": r["raw_response"], "failures": [k for k, v in r.items() if k in ["has_preview_env", "has_usepackage", "repetition_loop", "compile_ok"] and (v if k != "compile_ok" else not v)]})
        structured[label] = {"compile_rate": sum(1 for r in results if r["compile_ok"]) / n, "substantive_rate": sum(1 for r in results if r["true_substantive"]) / n, "bad_pattern_pass_rate": sum(1 for r in results if r["bad_patterns_pass"]) / n, "preview_environment_rate": sum(1 for r in results if r["has_preview_env"]) / n, "assistant_usepackage_rate": sum(1 for r in results if r["has_usepackage"]) / n, "assistant_documentclass_rate": sum(1 for r in results if r["has_documentclass"]) / n, "decorations_geometric_rate": sum(1 for r in results if r["has_decorations_geometric"]) / n, "repetition_loop_rate": sum(1 for r in results if r["repetition_loop"]) / n, "closing_fence_exactly_once_rate": sum(1 for r in results if r["closing_fence_exactly_once"]) / n, "avg_code_length": label_avg_len, "avg_code_length_ratio_vs_base": label_avg_len / base_avg_len if base_avg_len > 0 else 1.0}
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
    args = parser.parse_args()
    run_ab_eval(args.config, args.dataset, args.eval_manifest, args.sentinel_manifest, args.adapter_path, args.checkpoint_dir, args.num_samples, args.seed, args.max_tokens, args.skip_base, args.skip_finetuned, args.out_dir)

if __name__ == "__main__":
    main()
