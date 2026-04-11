import mlx.core as mx
import time
from fused_ops_mlx.infra.ops.rms_norm import fused_rms_norm

def benchmark_component(func, *args, name="Component", iterations=100):
    # Warmup
    for _ in range(10):
        func(*args)
    mx.eval(*args)
    
    start = time.perf_counter()
    for _ in range(iterations):
        res = func(*args)
        mx.eval(res)
    end = time.perf_counter()
    
    avg_ms = (end - start) * 1000 / iterations
    print(f"🚀 {name}: {avg_ms:.4f} ms")
    return avg_ms

def run_mega_benchmark():
    print("📊 MNPP Mega-Build Component Benchmark (M-series Native)")
    
    batch, length, dim = 4, 1024, 4096
    x = mx.random.normal((batch, length, dim))
    weight = mx.ones((dim,))

    # 1. Standard Reference
    def ref_norm(x, weight):
        rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-5)
        return (x / rms) * weight

    # 2. MNPP Fused Norm
    print(f"\nTask: RMSNorm for {batch}x{length}x{dim} block")
    
    t1 = benchmark_component(ref_norm, x, weight, name="Standard MLX Norm")
    t2 = benchmark_component(fused_rms_norm, x, weight, name="MNPP Fused Norm")
    
    improvement = (t1 - t2) / t1 * 100
    print(f"\n🔥 Result: MNPP Norm Kernel is {improvement:.2f}% faster than standard.")

    # 3. SwiGLU Benchmark
    from fused_ops_mlx.infra.ops.swiglu import fused_swiglu
    x_gate = mx.random.normal((batch, length, dim))
    x_up = mx.random.normal((batch, length, dim))
    
    print(f"\nTask: SwiGLU for {batch}x{length}x{dim} block")
    
    def ref_swiglu(x, y):
        return (x * mx.sigmoid(x)) * y
    
    t3 = benchmark_component(ref_swiglu, x_gate, x_up, name="Standard MLX SwiGLU")
    t4 = benchmark_component(fused_swiglu, x_gate, x_up, name="MNPP Fused SwiGLU")
    
    improvement_swiglu = (t3 - t4) / t3 * 100
    print(f"\n🔥 Result: MNPP SwiGLU Kernel is {improvement_swiglu:.2f}% faster than standard.")

if __name__ == "__main__":
    run_mega_benchmark()
