from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .schemas import CompileStatus, CompileSummary
from .settings import CompilerConfig


def _dedupe_preserve_order(values: list[object]) -> list[object]:
    seen: set[object] = set()
    ordered: list[object] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def parse_tectonic_log(log_text: str) -> tuple[list[str], list[int], list[str]]:
    key_errors = [
        str(item)
        for item in _dedupe_preserve_order(
            [match.strip() for match in re.findall(r"^! (.+)$", log_text, flags=re.MULTILINE)]
        )
    ]
    line_hints = [
        int(item)
        for item in _dedupe_preserve_order([int(match) for match in re.findall(r"l\.(\d+)", log_text)])
    ]
    missing_packages = [
        str(item)
        for item in _dedupe_preserve_order(re.findall(r"File [`']([^`']+\.sty)['`] not found", log_text))
    ]
    return key_errors, line_hints, missing_packages


class CompilerService:
    def __init__(self, config: CompilerConfig):
        self.config = config

    def is_available(self) -> bool:
        return shutil.which(str(self.config.tectonic_binary)) is not None

    def compile_document(
        self,
        latex_source: str,
        output_dir: str | Path | None = None,
        job_name: str = "document",
    ) -> CompileSummary:
        if not self.is_available():
            return CompileSummary(
                status=CompileStatus.TOOL_MISSING,
                return_code=None,
                key_errors=["tectonic binary not found"],
                line_hints=[],
                missing_packages=[],
                stdout="",
                stderr="",
                log_text="",
                elapsed_seconds=0.0,
            )

        work_dir = Path(output_dir).resolve() if output_dir else Path(tempfile.mkdtemp(prefix="tikz_compile_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        tex_path = work_dir / f"{job_name}.tex"
        pdf_path = work_dir / f"{job_name}.pdf"
        log_path = work_dir / f"{job_name}.log"
        tex_path.write_text(latex_source, encoding="utf-8")

        command = [
            str(self.config.tectonic_binary),
            "-X",
            "compile",
            tex_path.name,
            "--outdir",
            ".",
        ]
        if self.config.untrusted:
            command.append("--untrusted")
        if self.config.keep_logs:
            command.append("--keep-logs")
        if self.config.keep_intermediates:
            command.append("--keep-intermediates")

        start = time.monotonic()
        try:
            process = subprocess.Popen(
                command,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            return CompileSummary(
                status=CompileStatus.TOOL_MISSING,
                return_code=None,
                key_errors=[f"failed to launch tectonic: {exc}"],
                line_hints=[],
                missing_packages=[],
                stdout="",
                stderr="",
                log_text="",
                elapsed_seconds=time.monotonic() - start,
                tex_path=tex_path,
                pdf_path=None,
                log_path=log_path if log_path.exists() else None,
                working_dir=work_dir,
            )

        try:
            stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            stdout, stderr = self._terminate_process_group(process)
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            key_errors, line_hints, missing_packages = parse_tectonic_log(log_text)
            return CompileSummary(
                status=CompileStatus.TIMEOUT,
                return_code=None,
                key_errors=key_errors or ["tectonic compilation timed out"],
                line_hints=line_hints,
                missing_packages=missing_packages,
                stdout=stdout,
                stderr=stderr,
                log_text=log_text,
                elapsed_seconds=time.monotonic() - start,
                tex_path=tex_path,
                pdf_path=pdf_path if pdf_path.exists() else None,
                log_path=log_path if log_path.exists() else None,
                working_dir=work_dir,
            )

        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        key_errors, line_hints, missing_packages = parse_tectonic_log(log_text)

        if return_code == 0 and pdf_path.exists():
            status = CompileStatus.SUCCESS
        elif key_errors:
            status = CompileStatus.FATAL_ERROR
        else:
            status = CompileStatus.RECOVERABLE_ERROR

        return CompileSummary(
            status=status,
            return_code=return_code,
            key_errors=key_errors,
            line_hints=line_hints,
            missing_packages=missing_packages,
            stdout=stdout,
            stderr=stderr,
            log_text=log_text,
            elapsed_seconds=time.monotonic() - start,
            tex_path=tex_path,
            pdf_path=pdf_path if pdf_path.exists() else None,
            log_path=log_path if log_path.exists() else None,
            working_dir=work_dir,
        )

    def _terminate_process_group(self, process: subprocess.Popen[str]) -> tuple[str, str]:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            return process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                return process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    return process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    return "", ""
