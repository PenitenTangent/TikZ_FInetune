from __future__ import annotations

import atexit
import inspect
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grammar import TikzGrammarState
from .mlx_runtime import MlxRuntimeUnavailableError, import_mlx_core
from .prompting import build_gemma_messages
from .schemas import GenerationRequest, GenerationResult
from .settings import MemoryConfig, ModelConfig
from .quarantine import assert_not_quarantined


class MlxDependencyError(RuntimeError):
    """Raised when MLX or mlx-vlm is not installed."""


@dataclass(slots=True)
class LoadedModel:
    model: Any
    processor: Any


_TEMP_ADAPTER_DIRS: list[Path] = []


def _cleanup_temp_adapters() -> None:
    for path in _TEMP_ADAPTER_DIRS:
        try:
            if path.exists():
                shutil.rmtree(path)
        except OSError:
            pass


atexit.register(_cleanup_temp_adapters)


def prepare_adapter_for_mlx_vlm(adapter_path: str | Path | None) -> Path | None:
    """Ensure adapter_path is a directory that mlx-vlm understands.

    If it's a file, creates a temporary directory with symlinks for:
    - adapter_config.json (from parent)
    - adapter.safetensors (the file itself)
    """
    if adapter_path in (None, ""):
        return None
    path = Path(adapter_path).expanduser().resolve()
    if path.is_dir():
        # Check if it has the required files
        if (path / "adapter_config.json").exists() and (path / "adapter.safetensors").exists():
            assert_not_quarantined(path / "adapter.safetensors")
            return path
        # If it has the config but the weights are named differently (e.g. clean_adapter.safetensors)
        # then we still need to fix it.
        pass

    # Find the config file
    parent_dir = path if path.is_dir() else path.parent
    config_path = parent_dir / "adapter_config.json"
    if not config_path.exists():
        # Try to find it in the grandparent if this is a checkpoint subdir
        if (parent_dir.parent / "adapter_config.json").exists():
            config_path = parent_dir.parent / "adapter_config.json"

    # If we still don't have a config, we can't do much
    if not config_path.exists():
        return path if path.is_dir() else parent_dir

    # Create a temp directory for mlx-vlm to use
    temp_dir = Path(tempfile.mkdtemp(prefix="mlx_adapter_"))
    _TEMP_ADAPTER_DIRS.append(temp_dir)
    
    print(f"Created temporary adapter shim at {temp_dir}")
    
    try:
        # Symlink the config
        os.symlink(config_path, temp_dir / "adapter_config.json")
        # Symlink the weights as adapters.safetensors
        weights_file = path if path.is_file() else (path / "adapters.safetensors")
        if not weights_file.exists():
            # Try to find ANY safetensors file in the directory
            safetensors_files = list(parent_dir.glob("*.safetensors"))
            if safetensors_files:
                weights_file = safetensors_files[0]
        
        if weights_file.exists():
            assert_not_quarantined(weights_file)
            os.symlink(weights_file, temp_dir / "adapters.safetensors")
    except OSError:
        # Fallback to copy if symlink fails
        shutil.copy2(config_path, temp_dir / "adapter_config.json")
        if weights_file.exists():
            assert_not_quarantined(weights_file)
            shutil.copy2(weights_file, temp_dir / "adapters.safetensors")

    return temp_dir


def clear_mlx_cache() -> None:
    try:
        mx = import_mlx_core()
    except (ImportError, MlxRuntimeUnavailableError):
        return
    mx.clear_cache()


def configure_wired_limit(memory_config: MemoryConfig) -> None:
    """Set the MLX wired memory limit based on system recommendations and config.

    Calculates a safe 'wired' memory floor to prevent OS disk thrashing during 
    large model forward/backward passes on Apple Silicon.
    """
    try:
        mx = import_mlx_core()
    except (ImportError, MlxRuntimeUnavailableError):
        return

    if not hasattr(mx, "set_wired_limit") or not hasattr(mx, "device_info"):
        return

    info = mx.device_info()
    recommended = info.get("max_recommended_working_set_size")
    if not recommended:
        return

    configured_cap = int(memory_config.peak_memory_abort_gb * (1024**3))
    recommended_cap = int(recommended * memory_config.wired_limit_fraction)
    wired_limit = min(configured_cap, recommended_cap)
    if wired_limit <= 0:
        return
    mx.set_wired_limit(wired_limit)


class MlxVlmAdapter:
    """A high-level wrapper for the mlx-vlm inference engine.

    Handles model loading, prompt formatting (chat templates), and early-stopping 
    heuristics for TikZ generation.
    """
    def __init__(self, model_config: ModelConfig, memory_config: MemoryConfig):
        self.model_config = model_config
        self.memory_config = memory_config
        self.loaded: LoadedModel | None = None

    @staticmethod
    def load_model(model_id: str, adapter_path: str | Path | None = None) -> tuple[Any, Any]:
        """Load a model with an optional adapter, handling shimming if needed."""
        try:
            from mlx_vlm import load
        except ImportError:
            raise MlxDependencyError("mlx-vlm is not installed.")

        shimmed_path = prepare_adapter_for_mlx_vlm(adapter_path)
        if shimmed_path:
            print(f"Loading model {model_id} with adapter from {shimmed_path}...")
            return load(model_id, adapter_path=str(shimmed_path))
        else:
            print(f"Loading base model {model_id}...")
            return load(model_id)

    def _import_api(self) -> tuple[Any, Any, Any, Any]:
        try:
            import_mlx_core()
            from mlx_vlm import generate, load, stream_generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:
            raise MlxDependencyError(
                "mlx-vlm is not installed. Run `make install` before inference or training."
            ) from exc
        return load, generate, stream_generate, apply_chat_template

    def ensure_loaded(self) -> LoadedModel:
        if self.loaded is not None:
            return self.loaded

        load, _, _, _ = self._import_api()
        configure_wired_limit(self.memory_config)
        model, processor = load(self.model_config.model_id)
        self.loaded = LoadedModel(model=model, processor=processor)
        return self.loaded

    def unload(self) -> None:
        self.loaded = None
        clear_mlx_cache()

    def _format_prompt(self, request: GenerationRequest) -> str:
        loaded = self.ensure_loaded()
        _, _, _, apply_chat_template = self._import_api()
        chat_template_kwargs = {"enable_thinking": self.model_config.enable_thinking}

        if request.messages is not None:
            return apply_chat_template(
                loaded.processor,
                loaded.model.config,
                request.messages,
                num_images=len(request.image_paths),
                chat_template_kwargs=chat_template_kwargs,
            )

        system_prompt = request.system_prompt
        messages = build_gemma_messages(
            user_text=request.description,
            image_paths=request.image_paths,
            system_prompt=system_prompt,
        )
        return apply_chat_template(
            loaded.processor,
            loaded.model.config,
            messages,
            num_images=len(request.image_paths),
            chat_template_kwargs=chat_template_kwargs,
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        loaded = self.ensure_loaded()
        _, _, stream_generate, _ = self._import_api()
        prompt = self._format_prompt(request)
        image_arg = request.image_paths or None

        call_kwargs: dict[str, Any] = {
            "model": loaded.model,
            "processor": loaded.processor,
            "prompt": prompt,
            "image": image_arg,
            "max_tokens": request.max_tokens if request.max_tokens is not None else self.model_config.max_output_tokens,
            "temperature": request.temperature if request.temperature is not None else self.model_config.temperature,
            "top_p": request.top_p if request.top_p is not None else self.model_config.top_p,
            "top_k": request.top_k if request.top_k is not None else self.model_config.top_k,
            "verbose": False,
        }
        if request.min_p is not None and self._supports_parameter(stream_generate, "min_p"):
            call_kwargs["min_p"] = request.min_p
        if request.repetition_penalty is not None and self._supports_parameter(stream_generate, "repetition_penalty"):
            call_kwargs["repetition_penalty"] = request.repetition_penalty

        gen = stream_generate(**call_kwargs)
        text = ""
        # Sliding-window loop detector: tracks last LOOP_WINDOW non-empty lines.
        # If the same line appears >= LOOP_REPEAT_THRESHOLD times within the window,
        # generation is aborted (matches plan §2.2 / Extended §1.2).
        LOOP_WINDOW = 5
        LOOP_REPEAT_THRESHOLD = 2
        recent_lines: list[str] = []
        loop_detected = False
        # 4-gram blocker: prevents sub-line phrase looping (plan Extended §1.2).
        # Tracks all 4-token n-grams seen so far; aborts if a new quad-gram was
        # already emitted earlier in the same generation.
        NGRAM_N = 4
        seen_ngrams: set[tuple[str, ...]] = set()
        ngram_block_triggered = False
        # Running token buffer for n-gram extraction across chunk boundaries.
        _ngram_token_buf: list[str] = []
        # Grammar-guided early stopping (plan Extended §4).
        grammar = TikzGrammarState()
        for chunk in gen:
            chunk_text = self._coerce_generation_text(chunk)
            text += chunk_text

            # Update sliding-window with any new non-empty lines from this chunk.
            for line in chunk_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                recent_lines.append(stripped)
                if len(recent_lines) > LOOP_WINDOW:
                    recent_lines.pop(0)
                if (
                    len(recent_lines) == LOOP_WINDOW
                    and recent_lines.count(recent_lines[-1]) >= LOOP_REPEAT_THRESHOLD
                ):
                    loop_detected = True
                    break

            if loop_detected:
                break

            # 4-gram blocking: extract tokens from this chunk and check for repeats.
            _ngram_token_buf.extend(chunk_text.split())
            while len(_ngram_token_buf) >= NGRAM_N:
                quad = tuple(_ngram_token_buf[:NGRAM_N])
                _ngram_token_buf.pop(0)
                if quad in seen_ngrams:
                    ngram_block_triggered = True
                    break
                seen_ngrams.add(quad)
            if ngram_block_triggered:
                break

            # Grammar-guided early stopping: abort on structural violations.
            grammar.feed(chunk_text)
            if grammar.should_abort:
                break

            # Stop-token detection: terminate as soon as a complete diagram is present.
            if text.count("```") >= 2:  # closing fence
                break
            if "\\end{document}" in text:
                break
            if "\\end{tikzpicture}" in text and "\\begin{tikzpicture}" in text:
                break

        # ── Post-generation normalization ─────────────────────────────────────
        # Apply the same multi-pass normalization that was used during training so
        # that generated output is immediately compilable and stylistically
        # consistent (float quantization, semicolon healing, default option strip,
        # tikzstyle modernization, etc.).
        try:
            from .normalize import normalize_tikz as _normalize_tikz
            normalized_text = _normalize_tikz(text)
        except Exception:
            # Normalization is best-effort; never let it break inference.
            normalized_text = text

        return GenerationResult(
            text=normalized_text,
            prompt=prompt,
            model_id=self.model_config.model_id,
            image_paths=list(request.image_paths),
        )

    @staticmethod
    def _coerce_generation_text(result: Any) -> str:
        if isinstance(result, str):
            return result

        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text

        if isinstance(result, dict):
            for key in ("text", "output", "response", "generated_text"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"]

        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, str):
                    return item

        scalar_item = getattr(result, "item", None)
        if callable(scalar_item):
            try:
                scalar = scalar_item()
            except Exception:
                scalar = None
            if isinstance(scalar, str):
                return scalar

        return str(result)

    @staticmethod
    def _supports_parameter(function: Any, parameter: str) -> bool:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return False
        return parameter in signature.parameters
