import mlx.core as mx
import os

def load_swiglu_kernel():
    path = os.path.join(os.path.dirname(__file__), "swiglu.metal")
    with open(path, "r") as f:
        source = f.read()
    
    return mx.fast.metal_kernel(
        name="swiglu",
        input_names=["x", "y"],
        output_names=["out"],
        source=source
    )

_swiglu_kernel = None

def fused_swiglu(x, y):
    global _swiglu_kernel
    if _swiglu_kernel is None:
        _swiglu_kernel = load_swiglu_kernel()
    
    output = _swiglu_kernel(
        inputs=[x, y],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        template=[("T", x.dtype)]
    )
    return output[0]
