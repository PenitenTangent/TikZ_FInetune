from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
import inspect
from pathlib import Path

from .compiler import CompilerService
from .debug_render import RasterizationError, build_debug_render
from .infer import InferenceEngine
from .model_io import clear_mlx_cache
from .schemas import CompileStatus, RefinementAttempt, RepairMode
from .settings import PipelineConfig


@dataclass(slots=True)
class RefinementRunResult:
    attempts: list[RefinementAttempt] = field(default_factory=list)
    final_code: str = ""
    final_status: CompileStatus = CompileStatus.RECOVERABLE_ERROR


class RefinementOrchestrator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.inference = InferenceEngine(config)
        self.compiler = CompilerService(config.compiler)

    def run(self, description: str, output_dir: str | Path) -> RefinementRunResult:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        attempts: list[RefinementAttempt] = []
        retries_used = 0

        initial_attempts = self._generate_initial_candidates(description, target_dir)
        attempts.extend(initial_attempts)
        attempt = self._select_best_attempt(initial_attempts)
        best_compiled_attempt = self._best_successful_attempt(initial_attempts)

        compile_repairs = 0
        while (
            attempt.compile_summary is not None
            and attempt.compile_summary.status != CompileStatus.SUCCESS
            and compile_repairs < self.config.inference.repair_max_retries
            and retries_used < self.config.inference.max_retries
        ):
            compile_repairs += 1
            retries_used += 1

            repair_attempts = self._generate_compile_repair_candidates(
                attempt.generated_code,
                attempt.compile_summary,
                target_dir,
                repair_round=compile_repairs,
            )
            attempts.extend(repair_attempts)
            attempt = self._select_best_attempt([attempt, *repair_attempts])
            best_compiled_attempt = self._best_successful_attempt([best_compiled_attempt, *repair_attempts])

        if (
            best_compiled_attempt is not None
            and best_compiled_attempt.compile_summary is not None
            and best_compiled_attempt.compile_summary.pdf_path is not None
            and self.config.inference.visual_refine_on_success
            and retries_used < self.config.inference.max_retries
        ):
            try:
                self._maybe_clear_cache()
                debug_image = build_debug_render(
                    best_compiled_attempt.compile_summary.pdf_path,
                    target_dir / "debug",
                    step_px=self.config.inference.debug_grid_step_px,
                )
            except (RasterizationError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                attempt = best_compiled_attempt
            else:
                visual_attempts = self._generate_visual_repair_candidates(
                    description,
                    best_compiled_attempt.generated_code,
                    debug_image,
                    target_dir,
                )
                attempts.extend(visual_attempts)
                retries_used += 1
                attempt = self._select_best_attempt([best_compiled_attempt, *visual_attempts])
                if self._is_successful(attempt):
                    best_compiled_attempt = attempt
        elif best_compiled_attempt is not None:
            attempt = best_compiled_attempt

        final_status = attempt.compile_summary.status if attempt.compile_summary else CompileStatus.RECOVERABLE_ERROR
        return RefinementRunResult(
            attempts=attempts,
            final_code=attempt.generated_code,
            final_status=final_status,
        )

    def _compile_attempt(self, attempt: RefinementAttempt, output_dir: Path) -> None:
        attempt.compile_summary = self.compiler.compile_document(
            attempt.generated_code,
            output_dir=output_dir,
            job_name="candidate",
        )

    def _generate_initial_candidates(self, description: str, target_dir: Path) -> list[RefinementAttempt]:
        count = self.config.inference.initial_candidates
        attempts: list[RefinementAttempt] = []
        for index in range(count):
            self._maybe_clear_cache()
            attempt = self.inference.generate_initial(description)
            self._compile_attempt(attempt, self._candidate_output_dir(target_dir, "attempt_0", index, count))
            attempts.append(attempt)
        return attempts

    def _generate_compile_repair_candidates(
        self,
        code: str,
        summary,
        target_dir: Path,
        repair_round: int,
    ) -> list[RefinementAttempt]:
        count = self.config.inference.compile_repair_candidates
        attempts: list[RefinementAttempt] = []
        for index in range(count):
            self._maybe_clear_cache()
            repair_fn = self.inference.repair_compile_failure
            if "repair_round" in inspect.signature(repair_fn).parameters:
                attempt = repair_fn(code, summary, repair_round=repair_round)
            else:
                attempt = repair_fn(code, summary)
            self._compile_attempt(
                attempt,
                self._candidate_output_dir(target_dir, f"attempt_compile_{repair_round}", index, count),
            )
            attempts.append(attempt)
        return attempts

    def _generate_visual_repair_candidates(
        self,
        description: str,
        code: str,
        debug_image: Path,
        target_dir: Path,
    ) -> list[RefinementAttempt]:
        count = self.config.inference.visual_repair_candidates
        attempts: list[RefinementAttempt] = []
        for index in range(count):
            self._maybe_clear_cache()
            attempt = self.inference.repair_visual_output(description, code, str(debug_image))
            attempt.debug_image_path = debug_image
            self._compile_attempt(
                attempt,
                self._candidate_output_dir(target_dir, "attempt_visual_1", index, count),
            )
            attempts.append(attempt)
        return attempts

    @staticmethod
    def _candidate_output_dir(target_dir: Path, stem: str, index: int, total: int) -> Path:
        if total <= 1:
            return target_dir / stem
        return target_dir / f"{stem}_candidate_{index}"

    def _select_best_attempt(self, candidates: list[RefinementAttempt]) -> RefinementAttempt:
        if not candidates:
            raise ValueError("At least one candidate attempt is required.")

        best_attempt = candidates[0]
        best_score = (*self._attempt_rank_key(best_attempt), 0)
        for index, attempt in enumerate(candidates[1:], start=1):
            score = (*self._attempt_rank_key(attempt), index)
            if score > best_score:
                best_attempt = attempt
                best_score = score
        return best_attempt

    def _best_successful_attempt(self, candidates: list[RefinementAttempt | None]) -> RefinementAttempt | None:
        successful = [candidate for candidate in candidates if candidate is not None and self._is_successful(candidate)]
        if not successful:
            return None
        return self._select_best_attempt(successful)

    @staticmethod
    def _attempt_rank_key(attempt: RefinementAttempt) -> tuple[int, int, int, int, int, int, float]:
        if attempt.compile_summary is None:
            return (-1, -10_000, -10_000, -10_000, -1, -10_000, -1_000_000.0)

        summary = attempt.compile_summary
        status_rank = {
            CompileStatus.SUCCESS: 5,
            CompileStatus.RECOVERABLE_ERROR: 4,
            CompileStatus.FATAL_ERROR: 3,
            CompileStatus.TIMEOUT: 2,
            CompileStatus.TOOL_MISSING: 1,
        }.get(summary.status, 0)
        mode_rank = {
            RepairMode.INITIAL_GENERATION: 0,
            RepairMode.COMPILE_REPAIR: 1,
            RepairMode.VISUAL_REPAIR: 2,
        }.get(attempt.mode, 0)
        if summary.status != CompileStatus.SUCCESS:
            mode_rank = 0

        return (
            status_rank,
            -len(summary.key_errors),
            -len(summary.missing_packages),
            -len(summary.line_hints),
            mode_rank,
            -len(attempt.generated_code),
            -summary.elapsed_seconds,
        )

    def _maybe_clear_cache(self) -> None:
        policy = getattr(
            self.config.memory,
            "retry_cache_policy",
            "clear" if self.config.memory.clear_cache_between_retries else "none",
        )
        if policy == "none":
            return
        if policy == "clear":
            clear_mlx_cache()
            return
        if policy == "reload":
            self.inference.adapter.unload()
            return
        raise ValueError(f"Unsupported memory.retry_cache_policy: {policy}")

    @staticmethod
    def _is_successful(attempt: RefinementAttempt) -> bool:
        return attempt.compile_summary is not None and attempt.compile_summary.status == CompileStatus.SUCCESS
