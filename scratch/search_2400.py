
import mlx.core as mx
from mlx_vlm import load

model_id = "mlx-community/gemma-4-e4b-it-6bit"
model, processor = load(model_id)

for name, module in model.named_modules():
    params = module.parameters()
    for p_name, p_val in params.items():
        if isinstance(p_val, mx.array):
            if 2400 in p_val.shape:
                print(f"MATCH: {name}.{p_name}: {p_val.shape}")
        elif isinstance(p_val, dict):
            # Recursively check dict (though named_modules handles nesting)
            pass
