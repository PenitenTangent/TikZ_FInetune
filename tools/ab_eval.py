#!/usr/bin/env python3
"""A/B Evaluation for TikZ Finetuning.

Compares base model vs finetuned model on a fixed set of examples.
Can also show progression across multiple checkpoints.

Usage (full A/B at end of training):
  python3 tools/ab_eval.py \
    --config configs/curriculum_stage5.yaml \
    --adapter-path runs/tikz_lora_adapter.safetensors \
    --num-samples 5 --seed 42

Usage (finetuned-only during training):
  python3 tools/ab_eval.py \
    --config configs/curriculum_stage5.yaml \
    --adapter-path runs/tikz_lora_adapter.safetensors \
    --num-samples 5 --seed 42 --skip-base

Usage (progression across checkpoints):
  python3 tools/ab_eval.py \
    --config configs/curriculum_stage5.yaml \
    --checkpoint-dir runs/ \
    --num-samples 5 --seed 42
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


def _resolve_adapter_dir(adapter_path: str) -> str:
    """Resolve adapter path to a directory that mlx_vlm.load() accepts.

    mlx_vlm expects adapter_path to be a directory containing:
      - adapter_config.json
      - adapters.safetensors (or any *.safetensors)

    If given a bare .safetensors file, we look for adapter_config.json
    in the same directory or parent, and return the directory path.
    """
    p = Path(adapter_path)

    # If it's already a directory, use it directly
    if p.is_dir():
        return str(p)

    # It's a .safetensors file — find the directory with adapter_config.json
    parent = p.parent

    # Check if adapter_config.json exists in the same directory
    config_file = parent / "adapter_config.json"
    if config_file.exists():
        # If the file is already named adapters.safetensors, just return parent
        if p.name == "adapters.safetensors":
            return str(parent)
        # Otherwise, create a symlink so mlx_vlm can find it
        import tempfile, shutil
        tmp_dir = Path(tempfile.mkdtemp(prefix="ab_eval_adapter_"))
        shutil.copy2(config_file, tmp_dir / "adapter_config.json")
        (tmp_dir / "adapters.safetensors").symlink_to(p.resolve())
        return str(tmp_dir)

    # No adapter_config.json found — create a minimal one
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="ab_eval_adapter_"))
    (tmp_dir / "adapters.safetensors").symlink_to(p.resolve())
    (tmp_dir / "adapter_config.json").write_text(
        json.dumps({"lora_layers": 16, "lora_targets": "all"}),
        encoding="utf-8",
    )
    return str(tmp_dir)


# ── Data loading ────────────────────────────────────────────────────────────


def _load_eval_pool(dataset_path: Path) -> list[dict]:
    """Load samples from the gold eval dataset."""
    pool: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue

            user_text = _extract_text(messages[0])
            ref_text = _extract_text(messages[1])
            if not user_text:
                continue

            pool.append({
                "row_index": row_index,
                "sample_id": rec.get("sample_id", f"row_{row_index}"),
                "prompt_text": user_text,
                "reference_code": ref_text or "",
            })
    return pool


def _extract_text(msg: dict) -> str | None:
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    return None


def _select_fixed_samples(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Always select the same N samples for a given seed."""
    if len(pool) < n:
        n = len(pool)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pool)), n))
    return [pool[i] for i in indices]


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
        
        # Early stopping: only on a CLOSING fence (2nd occurrence)
        if result.count("```") >= 2:
            break
        if "\\end{document}" in result:
            break
        if "\\end{tikzpicture}" in result and "\\begin{tikzpicture}" in result:
            break

    return result


def _raw_to_compilable(raw_response: str, prompt_text: str) -> str:
    """Convert raw model response to a compilable LaTeX document.

    Extracts code from markdown fences, then uses normalize_tikz to wrap
    in a proper standalone document with all required packages.
    """
    extracted = extract_latex_from_response(raw_response)
    # normalize_tikz handles: unwrapping existing documents, stripping
    # comments, adding article+preview preamble, closing \\end{document}
    return normalize_tikz(extracted)


def _has_repetition_loop(text: str) -> bool:
    return has_repetition_failure(text)


# ── Evaluation ──────────────────────────────────────────────────────────────


def _evaluate_samples(
    *,
    label: str,
    adapter_path: str | None,
    samples: list[dict],
    cfg: Any,
    compiler: CompilerService,
    max_tokens: int,
    out_dir: Path,
) -> list[dict]:
    """Generate and compile for each sample. Returns list of result dicts."""
    from mlx_vlm import load

    print(f"\n{'─' * 60}", flush=True)
    print(f"  Loading model: {label}", flush=True)
    print(f"  Adapter: {adapter_path or '(none — base model)'}", flush=True)
    print(f"{'─' * 60}", flush=True)

    # mlx_vlm.load expects adapter_path to be a DIRECTORY containing
    # adapter_config.json and adapters.safetensors (or *.safetensors).
    resolved_adapter = _resolve_adapter_dir(adapter_path) if adapter_path else None

    model, processor = load(
        cfg.model.model_id,
        adapter_path=resolved_adapter,
        processor_config={"trust_remote_code": True},
    )

    results = []
    for i, sample in enumerate(samples):
        raw = _generate(model, processor, cfg, sample["prompt_text"], max_tokens)
        # Extract telemetry BEFORE normalization
        bad_pats = check_bad_patterns(raw)
        subst_feats = substantive_features(raw)
        token_feats = token_distribution_features(raw)

        # Normalize with audit
        latex_only = extract_latex_from_response(raw)
        compilable, norm_audit = normalize_with_audit(latex_only)

        has_env = bool(ENV_RE.search(compilable))
        has_cmd = bool(CMD_RE.search(compilable))
        substantive = has_env and has_cmd

        # Save generated files
        sample_dir = out_dir / label / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
        (sample_dir / "compilable.tex").write_text(compilable, encoding="utf-8")

        compile_ok = False
        compile_status = "not_attempted"
        render_sanity_passed = False
        if substantive:
            summary = compiler.compile_document(
                compilable, output_dir=sample_dir, job_name="output"
            )
            compile_ok = summary.pdf_path is not None and Path(summary.pdf_path).exists()
            compile_status = summary.status.value
            
            if compile_ok and summary.pdf_path:
                sanity_res = check_render_sanity(summary.pdf_path)
                render_sanity_passed = sanity_res.get("sanity_passed", False)
                
        # Check for truncation (missing closing environment)
        truncated = False
        if has_env and not re.search(
            r"\\end\{(?:tikzpicture|tikz-cd|circuitikz|axis)\}", compilable
        ):
            truncated = True
        repetition_loop = _has_repetition_loop(raw) or _has_repetition_loop(compilable)
        closing_fence_exactly_once = raw.count("```") == 1

        results.append({
            "i": i,
            "sample_id": sample["sample_id"],
            "raw_length": len(raw),
            "code_length": len(compilable),
            "has_tikz_env": has_env,
            "has_tikz_cmds": has_cmd,
            "substantive": substantive,
            "compile_ok": compile_ok,
            "compile_status": compile_status,
            "render_sanity_passed": render_sanity_passed,
            "truncated": truncated,
            "repetition_loop": repetition_loop,
            "closing_fence_exactly_once": closing_fence_exactly_once,
            "compilable_code": compilable,
            "raw_response": raw,
            "bad_patterns_pass": bad_pats["pass"],
            "bad_pattern_violations": bad_pats["violations"],
            "substantive_features": subst_feats,
            "token_distribution": token_feats,
            "normalization_audit": norm_audit,
        })

        status = "✅ Compiled" if compile_ok else ("⚠️ Truncated" if truncated else "❌ Failed")
        print(f"  [{i+1}/{len(samples)}] {sample['sample_id'][:16]} — {status} ({len(compilable)} chars)", flush=True)

    # Unload model to free memory
    del model, processor
    try:
        import_mlx_core().clear_cache()
    except Exception:
        pass

    return results


# ── Report formatting ───────────────────────────────────────────────────────


def _format_report(
    *,
    timestamp: str,
    config_path: str,
    dataset_path: str,
    num_samples: int,
    seed: int,
    max_tokens: int,
    samples: list[dict],
    all_results: dict[str, list[dict]],
) -> str:
    """Build a human-readable dated report."""
    lines: list[str] = []
    sep = "═" * 70
    thin = "─" * 70

    lines.append(sep)
    lines.append(f"  TikZ A/B Evaluation Report")
    lines.append(f"  Date: {timestamp}")
    lines.append(f"  Config: {config_path}")
    lines.append(f"  Dataset: {dataset_path}")
    lines.append(f"  Samples: {num_samples} (seed={seed})")
    lines.append(f"  Max tokens: {max_tokens}")
    labels = list(all_results.keys())
    for label in labels:
        lines.append(f"  Variant: {label}")
    lines.append(sep)
    lines.append("")

    # Per-sample comparison
    for i, sample in enumerate(samples):
        lines.append(thin)
        lines.append(f"  Sample {i+1}/{num_samples}  [{sample['sample_id']}]")
        lines.append(thin)
        lines.append("")

        # Prompt (truncated to 200 chars)
        prompt_preview = sample["prompt_text"][:200].replace("\n", " ")
        lines.append(f"PROMPT: {prompt_preview}...")
        lines.append("")

        # Reference
        ref = sample["reference_code"]
        if ref:
            ref_preview = ref[:600]
            if len(ref) > 600:
                ref_preview += "\n  ... (truncated)"
            lines.append(f"REFERENCE ({len(ref)} chars):")
            for line in ref_preview.split("\n"):
                lines.append(f"  {line}")
            lines.append("")

        # Each variant
        for label, results in all_results.items():
            r = results[i]
            if r["compile_ok"]:
                status = "✅ Compiled"
            elif r["truncated"]:
                status = "⚠️ TRUNCATED"
            else:
                status = f"❌ {r['compile_status']}"

            lines.append(f"{label.upper()} — {status} ({r['code_length']} chars):")

            # Show the actual TikZ code (extracted body between begin/end document)
            code = r["compilable_code"]
            # Extract body between \begin{document} and \end{document}
            body_match = re.search(
                r"\\begin\{document\}\n?(.*?)\\end\{document\}",
                code,
                flags=re.DOTALL,
            )
            body = body_match.group(1).strip() if body_match else code.strip()
            body_preview = body[:800]
            if len(body) > 800:
                body_preview += "\n  ... (truncated in report, full code saved to disk)"
            for line in body_preview.split("\n"):
                lines.append(f"  {line}")
            lines.append("")

        lines.append("")

    # Summary table
    lines.append(sep)
    lines.append("  SUMMARY")
    lines.append(sep)
    lines.append("")

    # Header
    col_width = max(len(l) for l in labels) + 2
    header = f"{'Metric':<30}"
    for label in labels:
        header += f"{label:>{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    n = num_samples

    # Metrics rows
    for metric_name, metric_key in [
        ("Compiled", "compile_ok"),
        ("Has TikZ env", "has_tikz_env"),
        ("Has TikZ commands", "has_tikz_cmds"),
        ("Substantive", "substantive"),
        ("Truncated", "truncated"),
        ("Repetition loop", "repetition_loop"),
        ("One closing fence", "closing_fence_exactly_once"),
    ]:
        row = f"{metric_name:<30}"
        for label in labels:
            count = sum(1 for r in all_results[label] if r[metric_key])
            pct = count / n * 100 if n else 0
            row += f"{f'{count}/{n} ({pct:.0f}%)':>{col_width}}"
        lines.append(row)

    # Avg code length
    row = f"{'Avg code length':<30}"
    for label in labels:
        avg = sum(r["code_length"] for r in all_results[label]) / n if n else 0
        row += f"{f'{avg:.0f} chars':>{col_width}}"
    lines.append(row)

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ── Checkpoint discovery ────────────────────────────────────────────────────


def _discover_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Find iteration-numbered checkpoint files in a directory tree."""
    found: list[tuple[int, Path]] = []
    pattern = re.compile(r"^(\d+)_adapters\.safetensors$")
    for path in checkpoint_dir.rglob("*.safetensors"):
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda x: x[0])
    return found


# ── Main ────────────────────────────────────────────────────────────────────


def run_ab_eval(
    *,
    config_path: str,
    dataset_path: str | None = None,
    eval_manifest_path: str | None = None,
    adapter_path: str | None = None,
    checkpoint_dir: str | None = None,
    num_samples: int = 120,
    seed: int = 42,
    max_tokens: int = 2048,
    skip_base: bool = False,
    skip_finetuned: bool = False,
    out_dir: str | None = None,
) -> Path:
    """Run the A/B evaluation. Returns the output directory path."""
    cfg = load_config(config_path)
    compiler = CompilerService(cfg.compiler)

    # Resolve dataset
    selected_dataset_path = eval_manifest_path or dataset_path
    ds_path = Path(selected_dataset_path) if selected_dataset_path else cfg.training.gold_eval_dataset_path
    if ds_path is None or not ds_path.exists():
        raise RuntimeError(f"Gold eval dataset not found: {ds_path}")

    # Load samples
    pool = _load_eval_pool(ds_path)
    samples = _select_fixed_samples(pool, num_samples, seed)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directory
    if out_dir:
        output_root = Path(out_dir)
    else:
        output_root = Path("outputs") / f"ab_eval_{timestamp_file}"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}", flush=True)
    print(f"  TikZ A/B Evaluation", flush=True)
    print(f"  {timestamp}", flush=True)
    print(f"  Dataset: {ds_path} ({len(pool)} total, {len(samples)} selected)", flush=True)
    print(f"  Max tokens: {max_tokens} (truncation guard)", flush=True)
    print(f"{'═' * 60}", flush=True)

    all_results: dict[str, list[dict]] = {}

    # Discover adapters to evaluate
    adapters: list[tuple[str, str | None]] = []

    if not skip_base:
        adapters.append(("base", None))

    if checkpoint_dir and not skip_finetuned:
        checkpoints = _discover_checkpoints(Path(checkpoint_dir))
        for iteration, ckpt_path in checkpoints:
            adapters.append((f"iter_{iteration}", str(ckpt_path)))

    if adapter_path and not skip_finetuned:
        adapters.append(("finetuned", adapter_path))

    if not adapters:
        raise RuntimeError("No adapters to evaluate. Provide --adapter-path or --checkpoint-dir.")

    # Run each variant
    for label, a_path in adapters:
        results = _evaluate_samples(
            label=label,
            adapter_path=a_path,
            samples=samples,
            cfg=cfg,
            compiler=compiler,
            max_tokens=max_tokens,
            out_dir=output_root,
        )
        all_results[label] = results

    # Generate report
    report = _format_report(
        timestamp=timestamp,
        config_path=config_path,
        dataset_path=str(ds_path),
        num_samples=len(samples),
        seed=seed,
        max_tokens=max_tokens,
        samples=samples,
        all_results=all_results,
    )

    # Save report
    report_path = output_root / "report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save structured results
    structured = {
        "timestamp": timestamp,
        "config_path": config_path,
        "dataset_path": str(ds_path),
        "num_samples": len(samples),
        "seed": seed,
        "max_tokens": max_tokens,
        "sample_ids": [s["sample_id"] for s in samples],
    }
    for label, results in all_results.items():
        n = len(results)
        structured[label] = {
            "compile_rate": sum(1 for r in results if r["compile_ok"]) / n if n else 0,
            "render_sanity_passed_rate": sum(1 for r in results if r["render_sanity_passed"]) / n if n else 0,
            "substantive_rate": sum(1 for r in results if r["substantive"]) / n if n else 0,
            "truncation_rate": sum(1 for r in results if r["truncated"]) / n if n else 0,
            "repetition_loop_rate": sum(1 for r in results if r["repetition_loop"]) / n if n else 0,
            "closing_fence_exactly_once_rate": sum(1 for r in results if r["closing_fence_exactly_once"]) / n if n else 0,
            "avg_code_length": sum(r["code_length"] for r in results) / n if n else 0,
        }
    (output_root / "results.json").write_text(
        json.dumps(structured, indent=2), encoding="utf-8"
    )

    # Print report
    print("\n" + report, flush=True)
    print(f"\nFull report saved to: {report_path}", flush=True)
    print(f"Generated files saved to: {output_root}", flush=True)

    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="TikZ A/B Evaluation")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--dataset", default=None, help="Path to eval dataset (default: gold_eval from config)")
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help="Path to a fixed eval JSONL set. Alias for --dataset for recovery probes.",
    )
    parser.add_argument("--adapter-path", default=None, help="Path to finetuned adapter weights")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory to search for numbered checkpoints")
    parser.add_argument("--num-samples", type=int, default=120, help="Number of eval samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens (avoid truncation)")
    parser.add_argument("--skip-base", action="store_true", help="Skip base model evaluation (faster)")
    parser.add_argument("--skip-finetuned", action="store_true", help="Skip adapter/checkpoint evaluation and run base only")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    run_ab_eval(
        config_path=args.config,
        dataset_path=args.dataset,
        eval_manifest_path=args.eval_manifest,
        adapter_path=args.adapter_path,
        checkpoint_dir=args.checkpoint_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        max_tokens=args.max_tokens,
        skip_base=args.skip_base,
        skip_finetuned=args.skip_finetuned,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
