
import mlx.core as mx
from mlx_vlm import load

model_id = "mlx-community/gemma-4-e4b-it-6bit"
model, processor = load(model_id)

for name, module in model.named_modules():
    if hasattr(module, "weight"):
        # For QuantizedLinear, it might have input_dims and output_dims
        if hasattr(module, "input_dims"):
            print(f"{name}: {module.input_dims} -> {module.output_dims}")
        else:
            print(f"{name}: {module.weight.shape}")
