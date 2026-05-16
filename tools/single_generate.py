#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from tikz_mlx.model_io import MlxVlmAdapter  # noqa: E402
from tikz_mlx.settings import load_config  # noqa: E402
from tools.ab_eval import _resolve_adapter_dir  # noqa: E402


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _adapter_config_path(resolved_adapter_dir: str | None) -> Path | None:
    if resolved_adapter_dir is None:
        return None
    path = Path(resolved_adapter_dir) / "adapter_config.json"
    return path if path.exists() else None


def _generate(
    *,
    model: Any,
    processor: Any,
    cfg: Any,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
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
        "temperature": temperature,
        "top_p": 1.0,
        "verbose": False,
    }
    if "repetition_penalty" in inspect.signature(stream_generate).parameters:
        kwargs["repetition_penalty"] = 1.2

    result = ""
    for chunk in stream_generate(**kwargs):
        result += MlxVlmAdapter._coerce_generation_text(chunk)
        if result.count("```") >= 2:
            break
        if "\\end{document}" in result:
            break
        if "\\end{tikzpicture}" in result and "\\begin{tikzpicture}" in result:
            break
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Single deterministic TikZ generation for adapter fingerprinting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out")
    args = parser.parse_args()

    from mlx_vlm import load

    cfg = load_config(args.config)
    resolved_adapter = _resolve_adapter_dir(args.adapter_path) if args.adapter_path else None
    model, processor = load(
        cfg.model.model_id,
        adapter_path=resolved_adapter,
        processor_config={"trust_remote_code": True},
    )
    output = _generate(
        model=model,
        processor=processor,
        cfg=cfg,
        prompt_text=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")

    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    adapter_config = _adapter_config_path(resolved_adapter)
    metadata = {
        "fixed_prompt_text_sha256": _sha256_text(args.prompt),
        "decode_config": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": 1.0,
            "repetition_penalty": 1.2,
        },
        "output_sha256": _sha256_text(output),
        "first_300_chars": output[:300],
        "char_count": len(output),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "resolved_adapter_dir": resolved_adapter,
        "adapter_sha256": _sha256_file(adapter_path) if adapter_path and adapter_path.is_file() else None,
        "adapter_config_sha256": _sha256_file(adapter_config),
        "config_path": str(Path(args.config).expanduser().resolve()),
    }
    metadata_path = Path(args.metadata_out) if args.metadata_out else output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(output_path), "metadata": str(metadata_path), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
