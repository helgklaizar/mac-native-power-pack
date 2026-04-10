"""
test_metal_batch3.py
====================
Tests remaining 38 Metal MSL modules (batch 3).
"""
import sys, time, traceback, json
from pathlib import Path
from datetime import datetime

ROOT  = Path(__file__).parent
METAL = ROOT / "modules/Metal"

for mod_dir in METAL.iterdir():
    src = mod_dir / "src"
    if src.exists():
        sys.path.insert(0, str(src))

import mlx.core as mx

RESULTS = []

def run_test(name, fn):
    print(f"\n{'─'*58}")
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

# ─── Tests ─────────────────────────────────────────────────────────────────────

def test_lm_cache():
    from lm_cache_mlx.infra.ops.cache import lm_cache_lookup
    qh = mx.array([1, 2, 3, 4, 5], dtype=mx.uint32)
    ch = mx.array([3, 5, 7, 1, 9], dtype=mx.uint32)
    hit, idx = lm_cache_lookup(qh, ch, stream=mx.gpu)
    assert hit.shape == (5,)
    return hit

def test_lookahead():
    from lookahead_mlx.infra.ops.draft import lookahead_decode
    hidden = mx.random.normal([8, 32])
    table  = mx.random.normal([4, 32])
    ids, sc = lookahead_decode(hidden, table, ngram_size=4, stream=mx.gpu)
    assert ids.shape == (8,) and sc.shape == (8,)
    return ids

def test_mamba():
    from mamba_mlx.infra.ops.ssm import mamba_ssm_step
    d = 64
    x      = mx.random.normal([d])
    A      = mx.random.normal([d])
    B      = mx.random.normal([d])
    C      = mx.random.normal([d])
    h_prev = mx.zeros([d])
    h, y   = mamba_ssm_step(x, A, B, C, h_prev, stream=mx.gpu)
    assert h.shape == (d,) and y.shape == (d,)
    return h

def test_medusa():
    from medusa_mlx.infra.ops.heads import medusa_draft_heads
    dim    = 32
    hidden = mx.random.normal([dim])
    heads_w = mx.random.normal([4, dim, dim])
    out    = medusa_draft_heads(hidden, heads_w, stream=mx.gpu)
    assert out.shape == (4, dim)
    return out

def test_metal_scheduler():
    from metal_scheduler_mlx.infra.ops.schedule import metal_schedule_ops
    n       = 16
    costs   = mx.random.uniform(shape=[n])
    deps    = mx.zeros([n, n])
    ready, prio = metal_schedule_ops(costs, deps, stream=mx.gpu)
    assert ready.shape == (n,) and prio.shape == (n,)
    return ready

def test_mimo_flash():
    from mimo_flash_mlx.infra.ops.mimo import mimo_flash_attention
    shape = [8, 32]
    q1 = mx.random.normal(shape)
    q2 = mx.random.normal(shape)
    k  = mx.random.normal(shape)
    v  = mx.random.normal(shape)
    out = mimo_flash_attention(q1, q2, k, v, stream=mx.gpu)
    assert out.shape == tuple(shape)
    return out

def test_minference():
    from minference_mlx.infra.ops.sparse import minference_sparse_attn
    q = mx.random.normal([8, 32])
    k = mx.random.normal([16, 32])
    v = mx.random.normal([16, 32])
    out = minference_sparse_attn(q, k, v, n_keys=16, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_mla():
    from mla_mlx.infra.ops.compress import mla_compress_kv
    kv        = mx.random.normal([16, 64])
    down_proj = mx.random.normal([16, 64])
    out       = mla_compress_kv(kv, down_proj, stream=mx.gpu)
    assert out.shape == (16, 16)
    return out

def test_multi_agent():
    from multi_agent_orchestrator_mlx.infra.ops.route import agent_route
    tasks  = mx.random.normal([8, 32])
    agents = mx.random.normal([4, 32])
    assigned, scores = agent_route(tasks, agents, stream=mx.gpu)
    assert assigned.shape == (8,) and scores.shape == (8,)
    return assigned

def test_mxfp4():
    from mxfp4_mlx.infra.ops.dequant import mxfp4_dequantize
    group_size = 8
    n_groups   = 16
    packed     = mx.array([[0xAB] * 4] * n_groups, dtype=mx.uint8)
    scales     = mx.ones([n_groups]) * 0.1
    out        = mxfp4_dequantize(packed, scales, group_size=group_size, stream=mx.gpu)
    assert out.shape == (n_groups * group_size,)
    return out

def test_nsa_attention():
    from nsa_attention_mlx.infra.ops.nsa import nsa_attention
    q    = mx.random.normal([8, 32])
    k    = mx.random.normal([16, 32])
    v    = mx.random.normal([16, 32])
    mask = mx.ones([8, 16], dtype=mx.uint8)
    out  = nsa_attention(q, k, v, mask, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_omniquant():
    from omniquant_mlx.infra.ops.dequant import omniquant_dequant
    qw    = mx.array([[3, -2, 1, 0] * 8] * 4, dtype=mx.int8)
    alpha = mx.ones([4]) * 0.5
    beta  = mx.zeros([4])
    out   = omniquant_dequant(qw, alpha, beta, stream=mx.gpu)
    assert out.shape == (4, 32)
    return out

def test_paged_attention():
    from paged_attention_mlx.infra.ops.paged import paged_attention
    block_size = 4
    n_blocks   = 4
    dim        = 16
    q          = mx.random.normal([8, dim])
    kv_pool    = mx.random.normal([n_blocks * block_size, dim])
    block_table = mx.array([0, 1, 2, 3], dtype=mx.uint32)
    out         = paged_attention(q, kv_pool, block_table, block_size=block_size, stream=mx.gpu)
    assert out.shape == (8, dim)
    return out

def test_plan_explore():
    from plan_and_explore_mlx.infra.ops.ucb import plan_explore_score
    n       = 32
    q_vals  = mx.random.normal([n])
    visits  = mx.array([float(i+1) for i in range(n)])
    out     = plan_explore_score(q_vals, visits, total_visits=100.0, stream=mx.gpu)
    assert out.shape == (n,)
    return out

def test_polarquant():
    from polarquant_mlx.infra.ops.rotate import polarquant_rotate
    w       = mx.random.normal([8, 32])
    rot_mat = mx.random.normal([32, 32])
    out     = polarquant_rotate(w, rot_mat, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_pyramid_kv():
    from pyramid_kv_mlx.infra.ops.pool import pyramid_kv_pool
    kv  = mx.random.normal([16, 32])
    out = pyramid_kv_pool(kv, win_size=4, stream=mx.gpu)
    assert out.shape == (4, 32)
    return out

def test_qaq():
    from qaq_mlx.infra.ops.quant import qaq_adaptive_quant
    x    = mx.random.normal([8, 32])
    bits = mx.array([4, 4, 4, 4, 8, 8, 8, 8], dtype=mx.uint8)
    q, sc = qaq_adaptive_quant(x, bits, stream=mx.gpu)
    assert q.shape == (8, 32) and sc.shape == (8,)
    return q

def test_qjl():
    from qjl_quant_mlx.infra.ops.dequant import qjl_dequantize
    codes  = mx.array([[1, -1, 0, 2] * 4] * 8, dtype=mx.int8)
    scales = mx.ones([16]) * 0.5
    proj   = mx.random.normal([8, 16])
    out    = qjl_dequantize(codes, scales, proj, stream=mx.gpu)
    assert out.shape == (8, 8)
    return out

def test_quarot():
    from quarot_mlx.infra.ops.hadamard import quarot_hadamard_transform
    x   = mx.random.normal([4, 8])
    out = quarot_hadamard_transform(x, stream=mx.gpu)
    assert out.shape == (4, 8)
    return out

def test_radix_cache():
    from radix_cache_mlx.infra.ops.score import radix_prefix_score
    tokens     = mx.array([[1,2,3,4]]*8, dtype=mx.uint32)
    trie_keys  = mx.array([[1,2,3,4],[5,6,7,8],[1,2,0,0]], dtype=mx.uint32)
    out        = radix_prefix_score(tokens, trie_keys, stream=mx.gpu)
    assert out.shape == (8,)
    return out

def test_ring_attention():
    from ring_attention_mlx.infra.ops.ring import ring_attention_step
    q   = mx.random.normal([8, 32])
    k   = mx.random.normal([8, 32])
    v   = mx.random.normal([8, 32])
    acc = mx.zeros([8, 32])
    out = ring_attention_step(q, k, v, acc, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_rocket_kv():
    from rocket_kv_mlx.infra.ops.evict import rocket_kv_evict
    imp     = mx.random.uniform(shape=[64])
    rec     = mx.random.uniform(shape=[64])
    mask    = rocket_kv_evict(imp, rec, alpha=0.6, threshold=0.3, stream=mx.gpu)
    assert mask.shape == (64,) and mask.dtype == mx.uint8
    return mask

def test_sage_attention():
    from sage_attention_mlx.infra.ops.sage import sage_attention
    q        = mx.random.normal([8, 32])
    k        = mx.random.normal([16, 32])
    v        = mx.random.normal([16, 32])
    smooth_k = mx.random.normal([32])
    out      = sage_attention(q, k, v, smooth_k, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_search_decode():
    from search_decode_mlx.infra.ops.beam import beam_search_step
    n_beams    = 4
    vocab_size = 32
    log_probs  = mx.random.normal([n_beams, vocab_size])
    beam_scores = mx.zeros([n_beams])
    out         = beam_search_step(log_probs, beam_scores, stream=mx.gpu)
    assert out.shape == (n_beams, vocab_size)
    return out

def test_self_spec():
    from self_spec_mlx.infra.ops.draft import self_speculative_draft
    hidden   = mx.random.normal([8, 32])
    lm_head  = mx.random.normal([16, 32])
    out      = self_speculative_draft(hidden, lm_head, stream=mx.gpu)
    assert out.shape == (8, 16)
    return out

def test_speculative_decode():
    from speculative_decode_mlx.infra.ops.verify import speculative_verify
    draft  = mx.random.normal([8, 32])
    target = mx.random.normal([8, 32])
    accept = speculative_verify(draft, target, stream=mx.gpu)
    assert accept.shape == (8,) and accept.dtype == mx.uint8
    return accept

def test_spinquant():
    from spinquant_mlx.infra.ops.rotate import spinquant_rotate_quantize
    w     = mx.random.normal([8, 32])
    R     = mx.random.normal([32, 32])
    scale = mx.ones([32])
    out   = spinquant_rotate_quantize(w, R, scale, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_spotlight():
    from spotlight_attention_mlx.infra.ops.spotlight import spotlight_attention
    q   = mx.random.normal([8, 32])
    k   = mx.random.normal([16, 32])
    v   = mx.random.normal([16, 32])
    out = spotlight_attention(q, k, v, spotlight_scale=1.0, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_star_attention():
    from star_attention_mlx.infra.ops.star import star_attention
    q_a = mx.random.normal([8, 32])
    q_c = mx.random.normal([8, 32])
    k   = mx.random.normal([16, 32])
    v   = mx.random.normal([16, 32])
    out = star_attention(q_a, q_c, k, v, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_streaming():
    from streaming_mlx.infra.ops.stream import streaming_attn_step
    q  = mx.random.normal([8, 32])
    kw = mx.random.normal([16, 32])
    vw = mx.random.normal([16, 32])
    out = streaming_attn_step(q, kw, vw, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_super_weight():
    from super_weight_mlx.infra.ops.scale import super_weight_scale
    w   = mx.random.normal([128])
    out = super_weight_scale(w, threshold=0.5, stream=mx.gpu)
    assert out.shape == (128,)
    return out

def test_tiered_cache():
    from tiered_cache_mlx.infra.ops.promote import tiered_cache_promote
    scores = mx.random.uniform(shape=[64])
    tier   = tiered_cache_promote(scores, hot=0.7, evict=0.2, stream=mx.gpu)
    assert tier.shape == (64,) and tier.dtype == mx.uint8
    return tier

def test_titans():
    from titans_mlx.infra.ops.memory import titans_neural_memory_step
    dim   = 32
    W_mem = mx.random.normal([dim * dim])
    x     = mx.random.normal([dim])
    out   = titans_neural_memory_step(x, W_mem, lr=0.01, stream=mx.gpu)
    assert out.shape == (dim * dim,)
    return out

def test_tome():
    from tome_mlx.infra.ops.merge import tome_merge_tokens
    n, dim = 16, 32
    tokens     = mx.random.normal([n, dim])
    similarity = mx.random.uniform(shape=[n, n])
    keep       = 8
    out        = tome_merge_tokens(tokens, similarity, keep=keep, stream=mx.gpu)
    assert out.shape == (keep, dim)
    return out

def test_tri_attention():
    from tri_attention_mlx.infra.ops.tri import tri_attention
    q  = mx.random.normal([8, 32])
    k  = mx.random.normal([16, 32])
    v  = mx.random.normal([16, 32])
    v2 = mx.random.normal([16, 32])
    out = tri_attention(q, k, v, v2, stream=mx.gpu)
    assert out.shape == (8, 32)
    return out

def test_triton_bridge():
    from triton_bridge_mlx.infra.ops.bridge import triton_to_metal
    M, N = 8, 16
    buf  = mx.random.normal([M * N])
    out  = triton_to_metal(buf, M, N, stream=mx.gpu)
    assert out.shape == (M, N)
    return out

def test_verifiable_rewards():
    from verifiable_rewards_mlx.infra.ops.reward import compute_verifiable_reward
    n        = 32
    logprobs = mx.random.normal([n])
    correct  = mx.ones([n], dtype=mx.uint32)
    reward   = compute_verifiable_reward(logprobs, correct, stream=mx.gpu)
    assert reward.shape == (n,)
    return reward

# ─── Main ──────────────────────────────────────────────────────────────────────

TESTS = [
    ("lm-cache-mlx",                  test_lm_cache),
    ("lookahead-mlx",                 test_lookahead),
    ("mamba-mlx",                     test_mamba),
    ("medusa-mlx",                    test_medusa),
    ("metal-scheduler-mlx",           test_metal_scheduler),
    ("mimo-flash-mlx",                test_mimo_flash),
    ("minference-mlx",                test_minference),
    ("mla-mlx",                       test_mla),
    ("multi-agent-orchestrator-mlx",  test_multi_agent),
    ("mxfp4-mlx",                     test_mxfp4),
    ("nsa-attention-mlx",             test_nsa_attention),
    ("omniquant-mlx",                 test_omniquant),
    ("paged-attention-mlx",           test_paged_attention),
    ("plan-and-explore-mlx",          test_plan_explore),
    ("polarquant-mlx",                test_polarquant),
    ("pyramid-kv-mlx",                test_pyramid_kv),
    ("qaq-mlx",                       test_qaq),
    ("qjl-quant-mlx",                 test_qjl),
    ("quarot-mlx",                    test_quarot),
    ("radix-cache-mlx",               test_radix_cache),
    ("ring-attention-mlx",            test_ring_attention),
    ("rocket-kv-mlx",                 test_rocket_kv),
    ("sage-attention-mlx",            test_sage_attention),
    ("search-decode-mlx",             test_search_decode),
    ("self-spec-mlx",                 test_self_spec),
    ("speculative-decode-mlx",        test_speculative_decode),
    ("spinquant-mlx",                 test_spinquant),
    ("spotlight-attention-mlx",       test_spotlight),
    ("star-attention-mlx",            test_star_attention),
    ("streaming-mlx",                 test_streaming),
    ("super-weight-mlx",              test_super_weight),
    ("tiered-cache-mlx",              test_tiered_cache),
    ("titans-mlx",                    test_titans),
    ("tome-mlx",                      test_tome),
    ("tri-attention-mlx",             test_tri_attention),
    ("triton-bridge-mlx",             test_triton_bridge),
    ("verifiable-rewards-mlx",        test_verifiable_rewards),
]

def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═'*58}")
    print(f"  🔥  METAL MODULES BATCH-3 TEST")
    print(f"  {ts}  |  MLX {mx.__version__}  |  {mx.default_device()}")
    print(f"{'═'*58}")

    for name, fn in TESTS:
        run_test(name, fn)

    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    total  = len(RESULTS)

    print(f"\n{'═'*58}")
    print(f"  SUMMARY  |  {passed}/{total} passed  |  {failed} failed")
    print(f"{'═'*58}")
    for r in RESULTS:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {icon}  {r['module']:<42} {r['latency_ms']:>8.2f} ms")

    report = {
        "timestamp": ts,
        "mlx_version": mx.__version__,
        "summary": {"total": total, "passed": passed, "failed": failed},
        "results": RESULTS
    }
    out_path = ROOT / "test_metal_results_batch3.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  📄 Report → {out_path.name}")
    print(f"{'═'*58}\n")
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
