"""
Metal Modules Test Suite
========================
Tests all 6 Metal MSL kernels from modules/Metal/* + validates
MLX model inference (mlx-community/Llama-3.2-1B-Instruct-4bit).

Usage:
    python test_metal_modules.py
"""

import sys
import time
import traceback
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "modules/Metal/flash-attention-mlx/src"))
sys.path.insert(0, str(Path(__file__).parent / "modules/Metal/bitnet-mlx/src"))
sys.path.insert(0, str(Path(__file__).parent / "modules/Metal/block-sparse-mlx/src"))
sys.path.insert(0, str(Path(__file__).parent / "modules/Metal/aqlm-mlx/src"))
sys.path.insert(0, str(Path(__file__).parent / "modules/Metal/attention-fuse-mlx/src"))
sys.path.insert(0, str(Path(__file__).parent / "modules/Metal/attention-matching-mlx/src"))

import mlx.core as mx

RESULTS = []

# ─── helpers ──────────────────────────────────────────────────────────────────

def run_test(name: str, fn):
    """Execute one kernel test, measure latency, record outcome."""
    print(f"\n{'─'*50}")
    print(f"  TEST: {name}")
    print(f"{'─'*50}")
    t0 = time.perf_counter()
    try:
        result = fn()
        mx.eval(result if isinstance(result, mx.array) else result[0])
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  ✅  PASS  |  latency: {elapsed:.2f} ms")
        RESULTS.append({"module": name, "status": "PASS", "latency_ms": round(elapsed, 2), "error": None})
        return result
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        err = traceback.format_exc()
        print(f"  ❌  FAIL  |  {e}")
        RESULTS.append({"module": name, "status": "FAIL", "latency_ms": round(elapsed, 2), "error": str(e)})
        return None


# ─── kernel tests ─────────────────────────────────────────────────────────────

def test_flash_attention():
    from flash_attention_mlx.infra.ops.attention import flash_attention
    B, S, D = 2, 16, 64
    q = mx.random.normal([B * S, D])
    k = mx.random.normal([B * S, D])
    v = mx.random.normal([B * S, D])
    out = flash_attention(q, k, v, stream=mx.gpu)
    assert out.shape == (B * S, D), f"Shape mismatch: {out.shape}"
    return out


def test_bitnet_ternary():
    from bitnet_mlx.infra.ops.ternary import bitnet_linear
    batch, in_f, out_f = 8, 64, 32
    x = mx.random.normal([batch, in_f])
    w_tern = mx.array(
        [[1 if (i+j) % 3 == 0 else (-1 if (i+j) % 3 == 1 else 0)
          for j in range(in_f)] for i in range(out_f)],
        dtype=mx.int8
    )
    scales = mx.ones([out_f])
    out = bitnet_linear(x, w_tern, scales, stream=mx.gpu)
    assert out.shape == (batch, out_f), f"Shape mismatch: {out.shape}"
    return out


def test_block_sparse():
    from block_sparse_mlx.infra.ops.sparse_mm import block_sparse_mm
    BS = 8
    M, K, N = 2 * BS, 2 * BS, 2 * BS  # must be divisible by BS
    A = mx.random.normal([M, K])
    B = mx.random.normal([K, N])
    # block_indices: rows=active blocks, cols=[row_block, col_block]
    block_indices = mx.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mx.uint32)
    out = block_sparse_mm(A, B, block_indices, block_size=BS, stream=mx.gpu)
    assert out.shape == (M, N), f"Shape mismatch: {out.shape}"
    return out


def test_aqlm_dequantize():
    from aqlm_mlx.infra.ops.aqlm import dequantize_aqlm
    M, num_books, book_size, group_size = 64, 4, 16, 8
    codes = mx.array(
        [[i % book_size for _ in range(num_books)] for i in range(M)],
        dtype=mx.uint8
    )
    codebooks = mx.random.normal([num_books, book_size, group_size])
    out = dequantize_aqlm(codes, codebooks, stream=mx.gpu)
    assert out.shape == (M, group_size), f"Shape mismatch: {out.shape}"
    return out


def test_attention_fuse():
    from attention_fuse_mlx.infra.ops.fuse import fuse_attention_rope
    S, H, D = 8, 4, 16
    q = mx.random.normal([S, H, D])
    k = mx.random.normal([S, H, D])
    v = mx.random.normal([S, H, D])
    cos = mx.random.normal([S + 64, D])
    sin = mx.random.normal([S + 64, D])
    q_out, k_out, v_out = fuse_attention_rope(q, k, v, cos, sin, offset=0, stream=mx.gpu)
    assert q_out.shape == q.shape, f"Q shape mismatch: {q_out.shape}"
    return (q_out, k_out, v_out)


def test_attention_matching():
    from attention_matching_mlx.infra.ops.match import attention_matching
    scores = mx.random.uniform(low=0.0, high=1.0, shape=[256])
    mask = attention_matching(scores, threshold=0.5, stream=mx.gpu)
    assert mask.shape == (256,), f"Shape mismatch: {mask.shape}"
    assert mask.dtype == mx.uint8, f"dtype mismatch: {mask.dtype}"
    return mask


# ─── model inference test ─────────────────────────────────────────────────────

def test_mlx_model_inference():
    """Run Llama-3.2-1B-Instruct-4bit via mlx_lm and measure tokens/sec."""
    print(f"\n{'═'*50}")
    print("  MODEL INFERENCE: Llama-3.2-1B-Instruct-4bit")
    print(f"{'═'*50}")
    t0 = time.perf_counter()
    try:
        from mlx_lm import load, generate
        model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        print(f"  Loading {model_name} ...")
        model, tokenizer = load(model_name)
        t_load = time.perf_counter()
        print(f"  Model loaded in {(t_load - t0)*1000:.0f} ms")

        prompt = "Describe Flash Attention in one sentence."
        print(f"  Prompt: '{prompt}'")

        t_gen = time.perf_counter()
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=64,
            verbose=False
        )
        t_done = time.perf_counter()

        gen_ms = (t_done - t_gen) * 1000
        token_count = len(tokenizer.encode(response))
        tps = token_count / ((t_done - t_gen) + 1e-9)

        print(f"  Response: {response[:200]}")
        print(f"  ✅  PASS  |  gen: {gen_ms:.0f} ms | ~{tps:.1f} tok/s | tokens: {token_count}")

        RESULTS.append({
            "module": "MLX-Inference/Llama-3.2-1B-4bit",
            "status": "PASS",
            "latency_ms": round(gen_ms, 1),
            "tokens_per_sec": round(tps, 1),
            "token_count": token_count,
            "response_preview": response[:200],
            "error": None
        })
    except ImportError:
        print("  ⚠️  mlx_lm not installed — skipping model inference test")
        RESULTS.append({
            "module": "MLX-Inference/Llama-3.2-1B-4bit",
            "status": "SKIP",
            "latency_ms": None,
            "error": "mlx_lm not installed"
        })
    except Exception as e:
        gen_ms = (time.perf_counter() - t0) * 1000
        print(f"  ❌  FAIL  |  {e}")
        RESULTS.append({
            "module": "MLX-Inference/Llama-3.2-1B-4bit",
            "status": "FAIL",
            "latency_ms": round(gen_ms, 1),
            "error": str(e)
        })


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═'*50}")
    print("  🔥  METAL MODULES TEST SUITE")
    print(f"  {ts}")
    print(f"{'═'*50}")
    print(f"  MLX device: {mx.default_device()}")

    # Kernel tests
    run_test("flash-attention-mlx",       test_flash_attention)
    run_test("bitnet-mlx / ternary_mm",   test_bitnet_ternary)
    run_test("block-sparse-mlx",          test_block_sparse)
    run_test("aqlm-mlx / dequantize",     test_aqlm_dequantize)
    run_test("attention-fuse-mlx / rope", test_attention_fuse)
    run_test("attention-matching-mlx",    test_attention_matching)

    # Model inference
    test_mlx_model_inference()

    # ── summary ──────────────────────────────────────────────────────────────
    passed  = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed  = sum(1 for r in RESULTS if r["status"] == "FAIL")
    skipped = sum(1 for r in RESULTS if r["status"] == "SKIP")
    total   = len(RESULTS)

    print(f"\n{'═'*50}")
    print(f"  SUMMARY  |  {passed}/{total} passed  |  {failed} failed  |  {skipped} skipped")
    print(f"{'═'*50}")
    for r in RESULTS:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⚠️"}.get(r["status"], "?")
        extra = f"  {r['latency_ms']} ms" if r.get("latency_ms") else ""
        tps_str = f"  {r['tokens_per_sec']} tok/s" if r.get("tokens_per_sec") else ""
        print(f"  {icon} {r['module']}{extra}{tps_str}")

    # ── write JSON report ─────────────────────────────────────────────────────
    report = {
        "timestamp": ts,
        "summary": {"total": total, "passed": passed, "failed": failed, "skipped": skipped},
        "results": RESULTS
    }
    out_path = Path(__file__).parent / "test_metal_results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  📄 Report saved → {out_path.name}")
    print(f"{'═'*50}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
