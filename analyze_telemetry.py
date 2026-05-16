import json
from pathlib import Path

runs_dir = Path("outputs/hparam_sweeps/stage1_fresh_20260514_121247/runs")
stats = {}

for run_path in runs_dir.iterdir():
    if not run_path.is_dir() or run_path.name == "stage2_checkpoints":
        continue
    
    telemetry_file = run_path / "gradient_clip_telemetry.jsonl"
    if not telemetry_file.exists():
        stats[run_path.name] = {"error": "No telemetry file"}
        continue
        
    losses = []
    grad_norms = []
    clip_rates = []
    memories = []
    iters = []
    
    with open(telemetry_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                losses.append(data.get("train_loss", 0))
                grad_norms.append(data.get("avg_grad_norm", 0))
                clip_rates.append(data.get("clipped_step_rate", 0))
                memories.append(data.get("peak_memory_gb", 0))
                iters.append(data.get("iteration", 0))
            except:
                pass
                
    if iters:
        stats[run_path.name] = {
            "max_iter": max(iters),
            "max_memory": max(memories),
            "avg_grad_norm": sum(grad_norms) / len(grad_norms),
            "avg_clip_rate": sum(clip_rates) / len(clip_rates),
            "start_loss": losses[0],
            "end_loss": losses[-1],
            "num_records": len(iters)
        }
    else:
        stats[run_path.name] = {"error": "Empty telemetry file"}

print(json.dumps(stats, indent=2))
