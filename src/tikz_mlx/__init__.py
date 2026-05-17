"""TikZ MLX pipeline package."""

# --- Monkey-patch mlx_vlm LoRaLayer to fix 6-bit dimension mismatch ---
# The default LoRaLayer in mlx_vlm.trainer.lora incorrectly calculates 
# input_dims for 6-bit models using (weight.shape[1] * (32 // bits)), 
# which yields 480 * 5 = 2400 instead of the correct 2560.
# This MUST happen before any mlx_vlm modules are imported.
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_vlm.trainer.lora as mlx_lora
    _original_init = mlx_lora.LoRaLayer.__init__

    def _linear_lora_dims(linear):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            scales = getattr(linear, "scales", None)
            group_size = getattr(linear, "group_size", None)
            if scales is not None and group_size:
                input_dims = int(scales.shape[-1]) * int(group_size)
            else:
                input_dims = (input_dims * 32) // linear.bits
        return int(input_dims), int(output_dims)
    
    def _patched_lora_init(self, linear, rank, alpha=0.1, dropout=0.0):
        # Call original init but then fix the dimensions if they are wrong
        _original_init(self, linear, rank, alpha, dropout)

        input_dims, output_dims = _linear_lora_dims(linear)
        if self.A.shape != (input_dims, rank) or self.B.shape != (rank, output_dims):
            print(f" [LoRA Patch] Fixing dimensions for {type(linear).__name__}: "
                  f"A {self.A.shape} -> ({input_dims}, {rank}), "
                  f"B {self.B.shape} -> ({rank}, {output_dims})")

            import math
            std_dev = 1 / math.sqrt(rank)
            self.A = mx.random.uniform(
                low=-std_dev,
                high=std_dev,
                shape=(input_dims, rank),
            )
            self.B = mx.zeros((rank, output_dims))
            self.scale = alpha / rank
    
    mlx_lora.LoRaLayer.__init__ = _patched_lora_init
    
    # Also patch the reference in utils if it's already there or for when it gets imported
    try:
        import mlx_vlm.trainer.utils as mlx_utils
        mlx_utils.LoRaLayer = mlx_lora.LoRaLayer
    except ImportError:
        pass
        
    print("✓ Applied 6-bit LoRA dimension patch to mlx_vlm")
except Exception:
    # We might be in a non-training environment where mlx_vlm isn't installed yet
    pass

__all__ = ["__version__"]

__version__ = "0.1.0"
