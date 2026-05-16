import json
from pathlib import Path

sweep_root = Path("outputs/hparam_sweeps/stage1_short_sweep_20260515_085711")
runs_dir = sweep_root / "runs"
gates_dir = sweep_root / "gates"

for run_path in sorted(runs_dir.iterdir()):
    if not run_path.is_dir() or run_path.name == "stage2_checkpoints":
        continue
    
    name = run_path.name
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    
    # Telemetry
    telemetry_file = run_path / "gradient_clip_telemetry.jsonl"
    if telemetry_file.exists():
        records = []
        with open(telemetry_file) as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
        if records:
            first = records[0]
            last = records[-1]
            avg_norm = sum(r["avg_grad_norm"] for r in records) / len(records)
            avg_clip = sum(r["clipped_step_rate"] for r in records) / len(records)
            avg_scale = sum(r["avg_clip_scale"] for r in records) / len(records)
            losses = [r["train_loss"] for r in records]
            
            print(f"  Iterations:       {first['iteration']} → {last['iteration']} ({len(records)} checkpoints)")
            print(f"  Train Loss:       {losses[0]:.4f} → {losses[-1]:.4f} (min={min(losses):.4f})")
            print(f"  Avg Grad Norm:    {avg_norm:.1f}")
            print(f"  Avg Clip Rate:    {avg_clip:.3f}")
            print(f"  Avg Clip Scale:   {avg_scale:.4f}")
            print(f"  Peak Memory:      {last.get('peak_memory_gb', 'N/A')} GB")
            print(f"  Max Grad Norm:    {last.get('max_grad_norm', 'N/A')}")
            
            # Show loss trajectory in quarters
            q = len(losses)
            if q >= 4:
                q1, q2, q3, q4 = losses[q//4], losses[q//2], losses[3*q//4], losses[-1]
                print(f"  Loss trajectory:  Q1={q1:.3f}  Q2={q2:.3f}  Q3={q3:.3f}  Q4={q4:.3f}")
            
            # Check if clip rate decreased over time
            if len(records) >= 10:
                early_clip = sum(r["clipped_step_rate"] for r in records[:5]) / 5
                late_clip = sum(r["clipped_step_rate"] for r in records[-5:]) / 5
                print(f"  Clip rate trend:  early={early_clip:.3f} → late={late_clip:.3f}")
        else:
            print("  (empty telemetry)")
    else:
        print("  ⚠ No telemetry file found")
    
    # Coverage state
    for cs_file in run_path.glob("coverage_state*.json"):
        try:
            cs = json.loads(cs_file.read_text())
            print(f"  Coverage:         step {cs['global_step']}/{cs['target_steps']}")
        except:
            pass
    
    # Named checkpoints
    nc_dir = run_path / "named_checkpoints"
    if nc_dir.exists():
        for meta_file in sorted(nc_dir.glob("*.metadata.json")):
            try:
                meta = json.loads(meta_file.read_text())
                val_loss = meta.get("metrics", {}).get("validation_loss")
                step = meta.get("global_step", "?")
                print(f"  Checkpoint:       step={step}, val_loss={val_loss}")
            except:
                pass

    # Gate results
    variant_name = name.replace("sweep_", "")
    gate_dir = gates_dir / variant_name / "quick"
    if not gate_dir.exists():
        gate_dir = gates_dir / name / "quick"
    if not gate_dir.exists():
        # Try to find it
        for gd in gates_dir.iterdir():
            if gd.is_dir() and variant_name.replace("sweep_", "") in gd.name:
                gate_dir = gd / "quick"
                break
    
    grad_gate = gate_dir / "gradient_telemetry_gate.json" if gate_dir.exists() else None
    if grad_gate and grad_gate.exists():
        gate = json.loads(grad_gate.read_text())
        print(f"  Grad Gate:        passed={gate['passed']}")
        if not gate["passed"]:
            for f in gate.get("failures", []):
                print(f"                    ✗ {f}")
    else:
        print("  Grad Gate:        (not found)")

# Check manifest for overall status
manifest = sweep_root / "manifest.jsonl"
if manifest.exists():
    entries = []
    with open(manifest) as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except:
                    pass
    print(f"\n{'='*80}")
    print(f"  MANIFEST SUMMARY ({len(entries)} entries)")
    print(f"{'='*80}")
    for e in entries:
        status = e.get("status", "?")
        variant = e.get("variant_id", e.get("name", "?"))
        print(f"  {variant}: {status}")
