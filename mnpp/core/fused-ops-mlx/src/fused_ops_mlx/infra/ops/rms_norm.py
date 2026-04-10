import mlx.core as mx
import os

def load_kernel():
    # Load MSL source
    path = os.path.join(os.path.dirname(__file__), "rms_norm.metal")
    with open(path, "r") as f:
        source = f.read()
    
    # Create the JIT kernel
    return mx.fast.metal_kernel(
        name="rms_norm",
        input_names=["input", "weight", "D", "epsilon"],
        output_names=["output"],
        source=source
    )

_rms_norm_kernel = None

def fused_rms_norm(x, weight, eps=1e-5):
    global _rms_norm_kernel
    if _rms_norm_kernel is None:
        _rms_norm_kernel = load_kernel()
    
    # Shape checks and grid config
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    rows, d = x.shape
    
    # Grid: one thread per row
    grid = (rows, 1, 1)
    threadgroup = (1, 1, 1) # Simplistic first version
    
    # Execute kernel
    output = _rms_norm_kernel(
        inputs=[x, weight, d, eps],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=grid,
        threadgroup=threadgroup,
        template=[("T", x.dtype)] # Pass the template type
    )
    
    return output[0].reshape(orig_shape)
