from pathlib import Path

import tikz_mlx.refine as refine_module
from tikz_mlx.schemas import CompileStatus, CompileSummary, RepairMode, RefinementAttempt
from tikz_mlx.settings import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "lora_prod.yaml"


def _config():
    config = load_config(CONFIG_PATH)
    config.inference.visual_refine_on_success = True
    config.inference.initial_candidates = 1
    config.inference.compile_repair_candidates = 1
    config.inference.visual_repair_candidates = 1
    return config


def _summary(status: CompileStatus, output_dir: str | Path, *, include_pdf: bool = False) -> CompileSummary:
    target_dir = Path(output_dir)
    return CompileSummary(
        status=status,
        return_code=0 if status == CompileStatus.SUCCESS else 1,
        key_errors=[] if status == CompileStatus.SUCCESS else ["compile failed"],
        line_hints=[],
        missing_packages=[],
        stdout="",
        stderr="",
        log_text="",
        elapsed_seconds=0.01,
        pdf_path=target_dir / "candidate.pdf" if include_pdf else None,
        log_path=target_dir / "candidate.log",
        working_dir=target_dir,
    )


def test_compile_repair_recovers_before_visual_refine(monkeypatch, tmp_path: Path) -> None:
    class FakeInferenceEngine:
        def __init__(self, config):
            pass

        def generate_initial(self, description: str) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.INITIAL_GENERATION, "initial", "broken")

        def repair_compile_failure(self, code: str, summary: CompileSummary) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.COMPILE_REPAIR, "repair", "fixed")

        def repair_visual_output(self, description: str, code: str, debug_image_path: str) -> RefinementAttempt:
            raise AssertionError("visual repair should not run when disabled")

    class FakeCompilerService:
        def __init__(self, config):
            pass

        def compile_document(self, latex_source: str, output_dir=None, job_name: str = "candidate") -> CompileSummary:
            if latex_source == "fixed":
                return _summary(CompileStatus.SUCCESS, output_dir, include_pdf=True)
            return _summary(CompileStatus.RECOVERABLE_ERROR, output_dir)

    monkeypatch.setattr(refine_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(refine_module, "CompilerService", FakeCompilerService)

    config = _config()
    config.inference.visual_refine_on_success = False
    result = refine_module.RefinementOrchestrator(config).run("triangle", tmp_path)

    assert result.final_status == CompileStatus.SUCCESS
    assert result.final_code == "fixed"
    assert [attempt.mode for attempt in result.attempts] == [
        RepairMode.INITIAL_GENERATION,
        RepairMode.COMPILE_REPAIR,
    ]


def test_successful_visual_repair_becomes_final_candidate(monkeypatch, tmp_path: Path) -> None:
    class FakeInferenceEngine:
        def __init__(self, config):
            pass

        def generate_initial(self, description: str) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.INITIAL_GENERATION, "initial", "compiled")

        def repair_compile_failure(self, code: str, summary: CompileSummary) -> RefinementAttempt:
            raise AssertionError("compile repair should not run")

        def repair_visual_output(self, description: str, code: str, debug_image_path: str) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.VISUAL_REPAIR, "visual", "visual-fixed")

    class FakeCompilerService:
        def __init__(self, config):
            pass

        def compile_document(self, latex_source: str, output_dir=None, job_name: str = "candidate") -> CompileSummary:
            if latex_source in {"compiled", "visual-fixed"}:
                return _summary(CompileStatus.SUCCESS, output_dir, include_pdf=True)
            return _summary(CompileStatus.FATAL_ERROR, output_dir)

    monkeypatch.setattr(refine_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(refine_module, "CompilerService", FakeCompilerService)
    monkeypatch.setattr(
        refine_module,
        "build_debug_render",
        lambda pdf_path, output_dir, step_px=32: Path(output_dir) / "candidate.grid.png",
    )

    result = refine_module.RefinementOrchestrator(_config()).run("triangle", tmp_path)

    assert result.final_status == CompileStatus.SUCCESS
    assert result.final_code == "visual-fixed"
    assert [attempt.mode for attempt in result.attempts] == [
        RepairMode.INITIAL_GENERATION,
        RepairMode.VISUAL_REPAIR,
    ]


def test_failed_visual_repair_falls_back_to_last_good_candidate(monkeypatch, tmp_path: Path) -> None:
    class FakeInferenceEngine:
        def __init__(self, config):
            pass

        def generate_initial(self, description: str) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.INITIAL_GENERATION, "initial", "compiled")

        def repair_compile_failure(self, code: str, summary: CompileSummary) -> RefinementAttempt:
            raise AssertionError("compile repair should not run")

        def repair_visual_output(self, description: str, code: str, debug_image_path: str) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.VISUAL_REPAIR, "visual", "broken-visual")

    class FakeCompilerService:
        def __init__(self, config):
            pass

        def compile_document(self, latex_source: str, output_dir=None, job_name: str = "candidate") -> CompileSummary:
            if latex_source == "compiled":
                return _summary(CompileStatus.SUCCESS, output_dir, include_pdf=True)
            return _summary(CompileStatus.FATAL_ERROR, output_dir)

    monkeypatch.setattr(refine_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(refine_module, "CompilerService", FakeCompilerService)
    monkeypatch.setattr(
        refine_module,
        "build_debug_render",
        lambda pdf_path, output_dir, step_px=32: Path(output_dir) / "candidate.grid.png",
    )

    result = refine_module.RefinementOrchestrator(_config()).run("triangle", tmp_path)

    assert result.final_status == CompileStatus.SUCCESS
    assert result.final_code == "compiled"
    assert result.attempts[-1].mode == RepairMode.VISUAL_REPAIR
    assert result.attempts[-1].compile_summary.status == CompileStatus.FATAL_ERROR


def test_compile_repairs_respect_total_retry_budget(monkeypatch, tmp_path: Path) -> None:
    state = {"repair_calls": 0}

    class FakeInferenceEngine:
        def __init__(self, config):
            pass

        def generate_initial(self, description: str) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.INITIAL_GENERATION, "initial", "broken-0")

        def repair_compile_failure(self, code: str, summary: CompileSummary) -> RefinementAttempt:
            state["repair_calls"] += 1
            return RefinementAttempt(
                RepairMode.COMPILE_REPAIR,
                f"repair-{state['repair_calls']}",
                f"broken-{state['repair_calls']}",
            )

        def repair_visual_output(self, description: str, code: str, debug_image_path: str) -> RefinementAttempt:
            raise AssertionError("visual repair should not run")

    class FakeCompilerService:
        def __init__(self, config):
            pass

        def compile_document(self, latex_source: str, output_dir=None, job_name: str = "candidate") -> CompileSummary:
            return _summary(CompileStatus.RECOVERABLE_ERROR, output_dir)

    monkeypatch.setattr(refine_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(refine_module, "CompilerService", FakeCompilerService)

    config = _config()
    config.inference.max_retries = 1
    config.inference.repair_max_retries = 1
    config.inference.visual_refine_on_success = False
    result = refine_module.RefinementOrchestrator(config).run("triangle", tmp_path)

    assert state["repair_calls"] == 1
    assert len(result.attempts) == 2
    assert result.final_code == "broken-1"


def test_initial_best_of_n_prefers_compilable_candidate(monkeypatch, tmp_path: Path) -> None:
    state = {"initial_calls": 0}

    class FakeInferenceEngine:
        def __init__(self, config):
            pass

        def generate_initial(self, description: str) -> RefinementAttempt:
            state["initial_calls"] += 1
            code = ["broken-a", "compiled", "broken-b"][state["initial_calls"] - 1]
            return RefinementAttempt(RepairMode.INITIAL_GENERATION, "initial", code)

        def repair_compile_failure(self, code: str, summary: CompileSummary) -> RefinementAttempt:
            raise AssertionError("compile repair should not run")

        def repair_visual_output(self, description: str, code: str, debug_image_path: str) -> RefinementAttempt:
            raise AssertionError("visual repair should not run")

    class FakeCompilerService:
        def __init__(self, config):
            pass

        def compile_document(self, latex_source: str, output_dir=None, job_name: str = "candidate") -> CompileSummary:
            if latex_source == "compiled":
                return _summary(CompileStatus.SUCCESS, output_dir, include_pdf=True)
            return _summary(CompileStatus.RECOVERABLE_ERROR, output_dir)

    monkeypatch.setattr(refine_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(refine_module, "CompilerService", FakeCompilerService)

    config = _config()
    config.inference.initial_candidates = 3
    config.inference.visual_refine_on_success = False
    result = refine_module.RefinementOrchestrator(config).run("triangle", tmp_path)

    assert result.final_status == CompileStatus.SUCCESS
    assert result.final_code == "compiled"
    assert state["initial_calls"] == 3
    assert len(result.attempts) == 3


def test_compile_repair_best_of_n_prefers_successful_fix(monkeypatch, tmp_path: Path) -> None:
    state = {"repair_calls": 0}

    class FakeInferenceEngine:
        def __init__(self, config):
            pass

        def generate_initial(self, description: str) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.INITIAL_GENERATION, "initial", "broken")

        def repair_compile_failure(self, code: str, summary: CompileSummary) -> RefinementAttempt:
            state["repair_calls"] += 1
            repaired = ["still-broken", "fixed"][state["repair_calls"] - 1]
            return RefinementAttempt(RepairMode.COMPILE_REPAIR, "repair", repaired)

        def repair_visual_output(self, description: str, code: str, debug_image_path: str) -> RefinementAttempt:
            raise AssertionError("visual repair should not run")

    class FakeCompilerService:
        def __init__(self, config):
            pass

        def compile_document(self, latex_source: str, output_dir=None, job_name: str = "candidate") -> CompileSummary:
            if latex_source == "fixed":
                return _summary(CompileStatus.SUCCESS, output_dir, include_pdf=True)
            return _summary(CompileStatus.RECOVERABLE_ERROR, output_dir)

    monkeypatch.setattr(refine_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(refine_module, "CompilerService", FakeCompilerService)

    config = _config()
    config.inference.max_retries = 1
    config.inference.repair_max_retries = 1
    config.inference.compile_repair_candidates = 2
    config.inference.visual_refine_on_success = False
    result = refine_module.RefinementOrchestrator(config).run("triangle", tmp_path)

    assert result.final_status == CompileStatus.SUCCESS
    assert result.final_code == "fixed"
    assert state["repair_calls"] == 2
    assert len(result.attempts) == 3


def test_retry_cache_policy_reload_unloads_adapter(monkeypatch, tmp_path: Path) -> None:
    state = {"unload_calls": 0}

    class _Adapter:
        def unload(self):
            state["unload_calls"] += 1

    class FakeInferenceEngine:
        def __init__(self, config):
            self.adapter = _Adapter()

        def generate_initial(self, description: str) -> RefinementAttempt:
            return RefinementAttempt(RepairMode.INITIAL_GENERATION, "initial", "broken")

        def repair_compile_failure(self, code: str, summary: CompileSummary) -> RefinementAttempt:
            raise AssertionError("compile repair should not run")

        def repair_visual_output(self, description: str, code: str, debug_image_path: str) -> RefinementAttempt:
            raise AssertionError("visual repair should not run")

    class FakeCompilerService:
        def __init__(self, config):
            pass

        def compile_document(self, latex_source: str, output_dir=None, job_name: str = "candidate") -> CompileSummary:
            return _summary(CompileStatus.RECOVERABLE_ERROR, output_dir)

    monkeypatch.setattr(refine_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(refine_module, "CompilerService", FakeCompilerService)

    config = _config()
    config.inference.max_retries = 0
    config.inference.repair_max_retries = 0
    config.inference.visual_refine_on_success = False
    config.memory.retry_cache_policy = "reload"
    refine_module.RefinementOrchestrator(config).run("triangle", tmp_path)

    assert state["unload_calls"] == 1
