import subprocess

from tools.run_with_live_progress_tqdm import _parse_iter


def _resume_offset(path: str) -> str:
    result = subprocess.run(
        ["bash", "tools/resume_offset.sh", path],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_resume_offset_strips_leading_zeroes_for_numeric_checkpoints() -> None:
    assert _resume_offset("runs/curriculum_stage1/0001000_adapters.safetensors") == "1000"
    assert _resume_offset("runs/curriculum_stage4/0003000_adapters.safetensors") == "3000"


def test_resume_offset_falls_back_to_zero_for_manual_adapter_names() -> None:
    assert _resume_offset("runs/curriculum_stage4/warmup_adapter.safetensors") == "0"
    assert _resume_offset("runs/curriculum_stage1/warmstart_stage1_3000.safetensors") == "0"


def test_progress_parser_ignores_training_header_but_accepts_global_resume_line() -> None:
    assert _parse_iter("Starting training..., scheduled batches: 14008", 16568) == (None, 16568)
    assert _parse_iter("Resuming from global iteration 2560 / 16568. Running 14008 remaining batches.", 16568) == (
        2560,
        16568,
    )
