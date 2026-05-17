
import mlx.core as mx
from mlx_vlm import load

model_id = "mlx-community/gemma-4-e4b-it-6bit"
model, processor = load(model_id)

print("MODEL CONFIG:")
print(model.config)

print("\nTOP LEVEL CHILDREN:")
for name, child in model.children().items():
    print(f"{name}: {type(child)}")
