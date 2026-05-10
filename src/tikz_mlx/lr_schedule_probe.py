from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def build_sft_joined_lr_schedule(
    optim: Any,
    *,
    peak_lr: float,
    total_steps: int,
    warmup_fraction: float = 0.1,
    cosine_end_fraction: float = 0.01,
) -> Callable[[int], Any]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    warmup_steps = max(0, int(total_steps * warmup_fraction))
    end_lr = peak_lr * cosine_end_fraction

    if warmup_steps > 0:
        linear = optim.linear_schedule(0.0, peak_lr, steps=warmup_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        cosine = optim.cosine_decay(peak_lr, cosine_steps, end=end_lr)
        joined = optim.join_schedules([linear, cosine], [warmup_steps])
    else:
        joined = optim.cosine_decay(peak_lr, total_steps, end=end_lr)

    def schedule(step: int) -> Any:
        if step >= total_steps:
            return end_lr
        return joined(max(0, step))

    return schedule


def lr_value_at_step(schedule: Callable[[int], Any], step: int) -> float:
    return _as_float(schedule(step))
