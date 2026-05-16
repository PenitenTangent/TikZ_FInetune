import json
import subprocess
from pathlib import Path


def test_gradient_telemetry_gate_reports_non_numeric_values(tmp_path: Path) -> None:
    telemetry = tmp_path / "gradient_clip_telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "iteration": 10,
                "train_loss": 1.0,
                "avg_grad_norm": 0.5,
                "avg_clip_scale": "bad",
                "clipped_step_rate": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["python3", "tools/check_gradient_telemetry.py", "--telemetry", str(telemetry)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "not numeric" in result.stderr
