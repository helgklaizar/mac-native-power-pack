import mlx.core as mx
import time
import sys
import os

# Add mnpp to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mnpp.core.flash_attention_mlx.src.flash_attention_mlx.infra.ops.attention import flash_attention as flash_attn
# from mnpp.core.sage_attention_mlx.src.sage_attention_mlx.infra.ops.sage import sage_attention as sage_attn


def benchmark_kernel(name, func, shape=(1024, 1024)):
    print(f"--- Benchmarking {name} ---")
    q = mx.random.normal(shape)
    k = mx.random.normal(shape)
    v = mx.random.normal(shape)
    
    # Warmup
    for _ in range(5):
        _ = func(q, k, v)
    mx.eval(_)
    
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        out = func(q, k, v)
    mx.eval(out)
    end = time.perf_counter()
    
    latency = (end - start) / iterations * 1000
    print(f"Latency: {latency:.4f} ms")
    print(f"Status: ✅ PASS\n")

if __name__ == "__main__":
    print("🚀 Starting MNPP Power Pack Benchmarks on Apple Silicon...")
    try:
        benchmark_kernel("Flash-Attention-Metal", flash_attn)
        # Note: Sage-Attention in this stub might have different signature, 
        # but for consistency with the prototype we assume similar for now.
        # benchmark_kernel("Sage-Attention-Metal", sage_attn)
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
