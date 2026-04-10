"""
test_metal_modules_extended.py
==============================
Tests the remaining 19 Metal MSL modules (все кроме первых 6).
Автоматически фиксирует результаты в test_metal_results_extended.json
"""

import sys, time, traceback, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
METAL = ROOT / "modules/Metal"

# Add all Metal module src dirs to path
for mod_dir in METAL.iterdir():
    src = mod_dir / "src"
    if src.exists():
        sys.path.insert(0, str(src))

import mlx.core as mx

RESULTS = []

def run_test(name, fn):
    print(f"\n{'─'*55}")
    print(f"  TEST: {name}")
    t0 = time.perf_counter()
    try:
        result = fn()
        first = result[0] if isinstance(result, (list, tuple)) else result
        mx.eval(first)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  ✅  PASS  |  {elapsed:.2f} ms")
        RESULTS.append({"module": name, "status": "PASS", "latency_ms": round(elapsed, 2), "error": None})
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        tb = traceback.format_exc().strip().splitlines()[-1]
        print(f"  ❌  FAIL  |  {tb}")
        RESULTS.append({"module": name, "status": "FAIL", "latency_ms": round(elapsed, 2), "error": str(e)})

# ── Test functions ─────────────────────────────────────────────────────────────

def test_content_prefix():
    from content_prefix_mlx.infra.ops.prefix import match_content_prefix
    q   = mx.random.normal([32, 64])
    ref = mx.random.normal([64])
    out = match_content_prefix(q, ref, stream=mx.gpu)
    assert out.shape == (32,)
    return out

def test_context_bridge():
    from context_bridge_mlx.infra.ops.bridge import context_bridge
    kv_a = mx.random.normal([128])
    kv_b = mx.random.normal([128])
    gate = mx.random.uniform(shape=[128])
    out  = context_bridge(kv_a, kv_b, gate, stream=mx.gpu)
    assert out.shape == (128,)
    return out

def test_context_engineering():
    from context_engineering_mlx.infra.ops.prune import score_and_prune
    tokens = mx.random.normal([16, 64])
    scores, mask = score_and_prune(tokens, threshold=0.0, stream=mx.gpu)
    assert scores.shape == (16,)
    assert mask.shape   == (16,)
    return scores, mask

def test_cuda_bridge():
    from cuda_bridge_mlx.infra.ops.layout import cuda_to_metal
    t = mx.random.normal([4, 32, 32, 3])
    out = cuda_to_metal(t, stream=mx.gpu)
    assert out.shape == t.shape
    return out

def test_eagle():
    from eagle_mlx.infra.ops.draft import eagle_draft
    hidden = mx.random.normal([8, 64])
    proj   = mx.random.normal([64, 64])
    out    = eagle_draft(hidden, proj, stream=mx.gpu)
    assert out.shape == (8, 64)
    return out

def test_evol_kv():
    from evol_kv_mlx.infra.ops.score import evol_kv_score
    weights = mx.random.uniform(shape=[4, 64])  # [n_heads, n_kv]
    out     = evol_kv_score(weights, stream=mx.gpu)
    assert out.shape == (64,)
    return out

def test_expert_selective():
    from expert_selective_mlx.infra.ops.gate import expert_select_topk
    logits  = mx.random.normal([16, 8])   # [seq, n_experts]
    idx, sc = expert_select_topk(logits, top_k=2, stream=mx.gpu)
    assert idx.shape == (16, 2)
    assert sc.shape  == (16, 2)
    return idx, sc

def test_flash_infer():
    from flash_infer_mlx.infra.ops.infer import flash_infer_decode
    page_size = 4
    n_pages   = 8
    dim       = 32
    q          = mx.random.normal([8, dim])
    kv_pages   = mx.random.normal([n_pages, page_size, dim])
    page_table = mx.array([0,1,2,3,4,5,6,7], dtype=mx.uint32)
    # reshape kv_pages to 2D for flat indexing in kernel
    kv_flat    = kv_pages.reshape([n_pages * page_size, dim])
    out        = flash_infer_decode(q, kv_flat.reshape([n_pages, page_size * dim]),
                                    page_table, page_size=page_size, stream=mx.gpu)
    assert out.shape == (8, dim)
    return out

def test_forge():
    from forge_mlx.infra.ops.forge import forge_layer_build
    x    = mx.random.normal([16, 64])
    w    = mx.random.normal([32, 64])
    bias = mx.zeros([32])
    out  = forge_layer_build(x, w, bias, stream=mx.gpu)
    assert out.shape == (16, 32)
    return out

def test_fused_moe():
    from fused_moe_mlx.infra.ops.moe import fused_moe
    seq, in_dim, out_dim, n_exp, topk = 8, 16, 16, 4, 2
    x          = mx.random.normal([seq, in_dim])
    w1         = mx.random.normal([n_exp, out_dim, in_dim])
    w2         = mx.random.normal([n_exp, out_dim, in_dim])
    expert_ids = mx.array([[0,1]]*seq, dtype=mx.uint32)
    gate_w     = mx.ones([seq, topk]) * 0.5
    out        = fused_moe(x, w1, w2, expert_ids, gate_w, stream=mx.gpu)
    assert out.shape == (seq, out_dim)
    return out

def test_h2o():
    from h2o_mlx.infra.ops.h2o import h2o_evict
    attn_sum = mx.random.uniform(shape=[64])
    keep     = h2o_evict(attn_sum, keep_ratio=0.5, stream=mx.gpu)
    assert keep.shape == (64,)
    assert keep.dtype == mx.uint8
    return keep

def test_hqq():
    from hqq_mlx.infra.ops.hqq import hqq_dequantize
    qw    = mx.array([[10, -5, 0, 3] * 4] * 4, dtype=mx.int8)  # (4, 16)
    scale = mx.ones([4]) * 0.1
    zero  = mx.zeros([4])
    out   = hqq_dequantize(qw, scale, zero, stream=mx.gpu)
    assert out.shape == (4, 16)
    return out

def test_itc():
    from inference_time_compute_mlx.infra.ops.itc import itc_scale_steps
    entropy = mx.random.uniform(shape=[32])
    steps   = itc_scale_steps(entropy, base=1.0, max_s=8.0, stream=mx.gpu)
    assert steps.shape == (32,)
    return steps

def test_infini_attention():
    from infini_attention_mlx.infra.ops.infini import infini_attention_update
    dim   = 64
    mem_M = mx.zeros([dim])
    mem_z = mx.zeros([dim])
    new_k = mx.random.normal([dim])
    new_v = mx.random.normal([dim])
    M_out, z_out = infini_attention_update(mem_M, mem_z, new_k, new_v, stream=mx.gpu)
    assert M_out.shape == (dim,)
    return M_out, z_out

def test_kvcomp():
    from kvcomp_mlx.infra.ops.compress import kv_compress
    kv  = mx.random.normal([16, 32])
    out = kv_compress(kv, ratio=2, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_kvtc():
    from kvtc_mlx.infra.ops.tiered import kvtc_route
    freq = mx.random.uniform(shape=[128])
    tier = kvtc_route(freq, hot=0.7, warm=0.3, stream=mx.gpu)
    assert tier.shape == (128,)
    assert tier.dtype == mx.uint8
    return tier

def test_layerskip():
    from layerskip_mlx.infra.ops.skip import layerskip_gate
    hidden = mx.random.normal([16, 64])
    exit_w = mx.random.normal([64])
    mask   = layerskip_gate(hidden, exit_w, threshold=0.0, stream=mx.gpu)
    assert mask.shape == (16,)
    return mask

def test_liger():
    from liger_kernel_mlx.infra.ops.liger import liger_cross_entropy
    logits = mx.random.normal([8, 32])
    labels = mx.array([0,1,2,3,4,5,6,7], dtype=mx.uint32)
    losses = liger_cross_entropy(logits, labels, stream=mx.gpu)
    assert losses.shape == (8,)
    return losses

def test_turboquant():
    from turboquant_mlx.infra.ops.quant import quantize, dequantize
    x     = mx.random.normal([8, 64])
    scale = mx.ones([8]) * 0.1
    zero  = mx.zeros([8])
    q     = quantize(x, scale, zero, stream=mx.gpu)
    assert q.shape == (8, 64) and q.dtype == mx.int8
    dq    = dequantize(q, scale, zero, stream=mx.gpu)
    assert dq.shape == (8, 64)
    return q, dq

# ── Main ──────────────────────────────────────────────────────────────────────

TESTS = [
    ("content-prefix-mlx",           test_content_prefix),
    ("context-bridge-mlx",           test_context_bridge),
    ("context-engineering-mlx",      test_context_engineering),
    ("cuda-bridge-mlx",              test_cuda_bridge),
    ("eagle-mlx",                    test_eagle),
    ("evol-kv-mlx",                  test_evol_kv),
    ("expert-selective-mlx",         test_expert_selective),
    ("flash-infer-mlx",              test_flash_infer),
    ("forge-mlx",                    test_forge),
    ("fused-moe-mlx",                test_fused_moe),
    ("h2o-mlx",                      test_h2o),
    ("hqq-mlx",                      test_hqq),
    ("inference-time-compute-mlx",   test_itc),
    ("infini-attention-mlx",         test_infini_attention),
    ("kvcomp-mlx",                   test_kvcomp),
    ("kvtc-mlx",                     test_kvtc),
    ("layerskip-mlx",                test_layerskip),
    ("liger-kernel-mlx",             test_liger),
    ("turboquant-mlx",               test_turboquant),
]

def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═'*55}")
    print(f"  🔥  METAL MODULES EXTENDED TEST SUITE")
    print(f"  {ts}  |  MLX {mx.__version__}  |  {mx.default_device()}")
    print(f"{'═'*55}")

    for name, fn in TESTS:
        run_test(name, fn)

    passed  = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed  = sum(1 for r in RESULTS if r["status"] == "FAIL")
    total   = len(RESULTS)

    print(f"\n{'═'*55}")
    print(f"  SUMMARY  |  {passed}/{total} passed  |  {failed} failed")
    print(f"{'═'*55}")
    for r in RESULTS:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {icon}  {r['module']:<38} {r['latency_ms']:>8.2f} ms")

    report = {
        "timestamp": ts,
        "mlx_version": mx.__version__,
        "summary": {"total": total, "passed": passed, "failed": failed},
        "results": RESULTS
    }
    out_path = ROOT / "test_metal_results_extended.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  📄 Report → {out_path.name}")
    print(f"{'═'*55}\n")

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
