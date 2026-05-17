import json
from pathlib import Path
import mlx.core as mx
from tikz_mlx.train import save_optimizer_state_sidecar, load_optimizer_state_sidecar

class LargeMockOptimizer:
    def __init__(self, size: int, value: float) -> None:
        # Generate nested dictionary with `size` distinct arrays
        # to exceed 1024 parameters and trigger the previous bug
        self.state = {
            f"layer_{i:04d}": {
                "weight": mx.array([value + float(i)]),
                "bias": mx.array([value + float(i) * 2.0])
            }
            for i in range(size)
        }

def main():
    tmp_dir = Path("scratch/test_verify")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_dir / "test_adapters.safetensors"
    
    # 1500 parameters will create 3000 distinct arrays in the state dictionary
    param_size = 1500
    print(f"Creating a large mock optimizer state with {param_size * 2} arrays (exceeding 1024)...")
    original_optimizer = LargeMockOptimizer(size=param_size, value=10.0)
    
    # Save the optimizer state sidecar
    print("Saving optimizer state sidecar using new safetensors implementation...")
    state_path = save_optimizer_state_sidecar(original_optimizer, checkpoint_path)
    print(f"Saved sidecar to: {state_path}")
    
    # Verify the saved files
    assert state_path is not None
    assert state_path.exists()
    assert state_path.name.endswith(".optimizer_state.safetensors")
    
    metadata_path = state_path.with_name(f"{state_path.name}.json")
    assert metadata_path.exists()
    
    # Verify we can load it back correctly
    print("Loading optimizer state sidecar back...")
    restored_optimizer = LargeMockOptimizer(size=param_size, value=0.0)
    load_optimizer_state_sidecar(restored_optimizer, state_path)
    
    # Verify that the restored arrays exactly match the original ones
    print("Validating restored array contents...")
    for i in range(param_size):
        orig_w = original_optimizer.state[f"layer_{i:04d}"]["weight"].tolist()
        rest_w = restored_optimizer.state[f"layer_{i:04d}"]["weight"].tolist()
        orig_b = original_optimizer.state[f"layer_{i:04d}"]["bias"].tolist()
        rest_b = restored_optimizer.state[f"layer_{i:04d}"]["bias"].tolist()
        assert orig_w == rest_w, f"Weight mismatch at layer {i}"
        assert orig_b == rest_b, f"Bias mismatch at layer {i}"
        
    print("\nSUCCESS! Saved and restored all 3000 arrays perfectly without any nanobind limitations!")

if __name__ == "__main__":
    main()
