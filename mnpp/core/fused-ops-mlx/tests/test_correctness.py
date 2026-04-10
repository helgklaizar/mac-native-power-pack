import mlx.core as mx
import numpy as np
from fused_ops_mlx.infra.ops.rms_norm import fused_rms_norm

def test_rms_norm_correctness():
    # Setup
    batch, length, dim = 2, 64, 1024
    x = mx.random.normal((batch, length, dim))
    weight = mx.ones((dim,))
    eps = 1e-5

    # 1. Standard MLX reference
    def ref_rms_norm(x, weight, eps):
        rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
        return (x / rms) * weight

    ref_out = ref_rms_norm(x, weight, eps)

    # 2. Our Fused Kernel
    fused_out = fused_rms_norm(x, weight, eps)

    # Compare
    diff = mx.abs(ref_out - fused_out).max().item()
    print(f"Max Absolute Difference: {diff}")
    
    if diff < 1e-5:
        print("✅ TEST PASSED: Fused RMSNorm matches reference.")
    else:
        print("❌ TEST FAILED: Accuracy mismatch.")

if __name__ == "__main__":
    test_rms_norm_correctness()
