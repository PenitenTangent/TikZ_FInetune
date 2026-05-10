#!/usr/bin/env python3
"""Live progress monitor using tqdm for a single-line progress bar with iter/s.

Usage:
  tools/run_with_live_progress_tqdm.py --label stage1 --total-iters 100 -- <command> ...

It mirrors raw output to stdout (only errors/warnings printed as separate lines),
and uses tqdm to display progress and `it/s` in the postfix.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import queue
import re
import time
from collections import deque
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

ITER_PATTERNS = (
    re.compile(r"\b(?:iter|iteration|step|steps)\b\D*(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE),
    re.compile(r"(\d+)\s*/\s*(\d+)", re.IGNORECASE),
)
SPEED_PATTERN = re.compile(r"\b(?:it/s|it/sec|iter/s|iter/sec|step/s|step/sec)[\s:=]*([0-9]+(?:\.[0-9]+)?)|([0-9]+(?:\.[0-9]+)?)\s*(?:it/s|it/sec|iter/s|iter/sec|step/s|step/sec)\b", re.IGNORECASE)
ERROR_HINT = re.compile(r"\b(error|traceback|exception|runtimeerror|importerror|valueerror)\b", re.IGNORECASE)
WARN_HINT = re.compile(r"\bwarning\b", re.IGNORECASE)


def _format_elapsed(seconds: float) -> str:
    secs = int(max(0, seconds))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_iter(line: str, fallback_total: int | None) -> tuple[int | None, int | None]:
    stripped = line.strip()
    # Skip JSON payloads like "iters": 1000
    if stripped.startswith('"') and stripped.endswith(","):
        return None, fallback_total

    for pat in ITER_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        cur = int(m.group(1))
        total = int(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else fallback_total
        return cur, total
    return None, fallback_total


def _parse_speed(line: str) -> float | None:
    m = SPEED_PATTERN.search(line)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    if val:
        return float(val)
    return None


def _scan_checkpoint_iter(checkpoint_dir: Path | None) -> int | None:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return None
    max_iter: int | None = None
    for path in checkpoint_dir.glob("*_adapters.safetensors"):
        stem = path.name.split("_", 1)[0]
        if stem.isdigit():
            value = int(stem)
            max_iter = value if max_iter is None else max(max_iter, value)
    return max_iter


def _emit_line(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def run_monitor(args: argparse.Namespace) -> int:
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("No command provided", file=sys.stderr)
        return 2

    if tqdm is None:
        print("\n" + "!" * 60, file=sys.stderr)
        print("WARNING: 'tqdm' is not installed in the current environment.", file=sys.stderr)
        print("The progress bar is disabled. To see it, run:", file=sys.stderr)
        print("    pip install tqdm", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)

    log_handle = None
    if getattr(args, "log_file", None):
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    q: queue.Queue[str | None] = queue.Queue()

    def reader():
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    start = time.time()
    cur_iter = None
    total_iter = args.total_iters
    checkpoint_dir = Path(args.checkpoint_dir) if getattr(args, "checkpoint_dir", None) else None
    iter_history: deque[tuple[float, int]] = deque(maxlen=10)
    parsed_speed = None
    peak_mem = None
    last_ckpt_scan = 0.0

    bar = None
    if tqdm is not None:
        bar = tqdm(
            desc=args.label,
            total=total_iter,
            file=sys.stdout,
            dynamic_ncols=True,
            disable=False,
            smoothing=0.1,
        )

    try:
        while True:
            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                item = None

            if item is None:
                if proc.poll() is not None:
                    break
                # no new line, just refresh display
                now = time.time()
                if checkpoint_dir is not None and (now - last_ckpt_scan) >= 1.0:
                    ckpt_iter = _scan_checkpoint_iter(checkpoint_dir)
                    if ckpt_iter is not None and (cur_iter is None or ckpt_iter > cur_iter):
                        cur_iter = ckpt_iter
                        iter_history.append((now, cur_iter))
                        if bar is not None:
                            try:
                                if total_iter is not None:
                                    bar.total = total_iter
                                # set absolute position (trainer emits absolute "iter X/Y")
                                bar.n = int(cur_iter)
                                # keep tqdm's last_print_n in sync so auto-refresh logic behaves
                                if hasattr(bar, "last_print_n"):
                                    try:
                                        bar.last_print_n = bar.n
                                    except Exception:
                                        pass
                                bar.refresh()
                            except Exception:
                                pass
                    last_ckpt_scan = now
                if bar is not None:
                    # compute smoothed speed
                    speed = None
                    if len(iter_history) >= 2:
                        t0, i0 = iter_history[0]
                        t1, i1 = iter_history[-1]
                        dt = max(1e-6, t1 - t0)
                        di = i1 - i0
                        if di >= 0:
                            speed = di / dt
                    display_speed = speed if speed is not None else parsed_speed
                    if display_speed is None:
                        display_speed = 0.0
                    display_speed = min(display_speed, 1000.0)
                    eta_txt = "n/a"
                    if display_speed and cur_iter is not None and total_iter is not None and total_iter > 0:
                        rem = max(0, total_iter - cur_iter)
                        eta = rem / display_speed if display_speed > 0 else None
                        if eta is not None:
                            eta_txt = _format_elapsed(eta)
                    bar.set_postfix({"it/s": f"{display_speed:.2f}", "eta": eta_txt})
                    bar.refresh()
                continue

            if item is None:
                # EOF marker
                break

            line = item.rstrip("\n")
            if log_handle is not None:
                log_handle.write(item)
                log_handle.flush()
            # Mirror only important lines (errors/warnings)
            if ERROR_HINT.search(line):
                _emit_line(line)
            elif args.show_warnings and WARN_HINT.search(line):
                _emit_line(line)

            found_iter, found_total = _parse_iter(line, total_iter)
            if found_iter is not None:
                cur_iter = found_iter
                if found_total is not None:
                    total_iter = found_total
                    if bar is not None:
                        bar.total = total_iter
                iter_history.append((time.time(), cur_iter))
                if bar is not None:
                    try:
                        if total_iter is not None:
                            bar.total = total_iter
                        bar.n = int(cur_iter)
                        if hasattr(bar, "last_print_n"):
                            try:
                                bar.last_print_n = bar.n
                            except Exception:
                                pass
                        bar.refresh()
                    except Exception:
                        pass

            maybe_sp = _parse_speed(line)
            if maybe_sp is not None:
                parsed_speed = maybe_sp

        exit_code = proc.wait()
        # finalize bar
        if bar is not None:
            # compute final display speed
            speed = None
            if len(iter_history) >= 2:
                t0, i0 = iter_history[0]
                t1, i1 = iter_history[-1]
                dt = max(1e-6, t1 - t0)
                di = i1 - i0
                if di >= 0:
                    speed = di / dt
            display_speed = speed if speed is not None else (parsed_speed if parsed_speed is not None else 0.0)
            display_speed = min(display_speed, 1000.0)
            bar.set_postfix({"it/s": f"{display_speed:.2f}"})
            bar.close()

        elapsed = time.time() - start
        if exit_code == 0:
            _emit_line(f"{args.label} complete (elapsed={_format_elapsed(elapsed)})")
        else:
            _emit_line(f"{args.label} failed (exit {exit_code})")
        return exit_code
    finally:
        if bar is not None and not bar.disable:
            try:
                bar.close()
            except Exception:
                pass
        if log_handle is not None:
            log_handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="train")
    parser.add_argument("--total-iters", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--show-warnings", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return run_monitor(args)


if __name__ == '__main__':
    raise SystemExit(main())
