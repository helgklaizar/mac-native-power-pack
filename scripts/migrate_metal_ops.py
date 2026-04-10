"""
migrate_metal_ops.py
====================
Автоматически мигрирует все Metal ops-файлы из старого API MLX
(full [[kernel]] declarations) в новый MLX 0.31.1+ body-only стиль.

Запускается один раз: python migrate_metal_ops.py
"""
import re
import sys
from pathlib import Path

METAL_ROOT = Path("/Users/k/Documents/PROJECTS/MVP/for-mac/modules/Metal")

# Новые body-only исходники для каждого модуля
# Ключ = имя файла ops/*.py, значение = полный новый контент файла
MIGRATIONS = {

# ── content-prefix-mlx ────────────────────────────────────────────────────────
"content-prefix-mlx/src/content_prefix_mlx/infra/ops/prefix.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi = thread_position_in_grid.x;
    uint ref_len = ref_len_arr[0];
    uint dim     = dim_arr[0];
    float score = 0.0f;
    for (uint d = 0; d < dim; ++d)
        score += query[qi * dim + d] * ref[d];
    out[qi] = score;
"""

def match_content_prefix(query: mx.array, ref: mx.array,
                          stream: mx.Stream = mx.gpu) -> mx.array:
    """Scores each query token against a prefix reference vector."""
    assert len(query.shape) == 2 and len(ref.shape) == 1
    seq_len, dim = query.shape
    kernel = mx.fast.metal_kernel(
        name="content_prefix_match",
        input_names=["query", "ref", "ref_len_arr", "dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[query, ref,
                mx.array([ref.shape[0]], dtype=mx.uint32),
                mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.float32],
        grid=(seq_len, 1, 1),
        threadgroup=(min(seq_len, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── context-bridge-mlx ────────────────────────────────────────────────────────
"context-bridge-mlx/src/context_bridge_mlx/infra/ops/bridge.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint idx = thread_position_in_grid.x;
    float g  = gate[idx];
    out[idx] = g * kv_a[idx] + (1.0f - g) * kv_b[idx];
"""

def context_bridge(kv_a: mx.array, kv_b: mx.array, gate: mx.array,
                   stream: mx.Stream = mx.gpu) -> mx.array:
    """Blends two KV context tensors using a learned gate vector."""
    assert kv_a.shape == kv_b.shape == gate.shape
    kernel = mx.fast.metal_kernel(
        name="context_bridge_kernel",
        input_names=["kv_a", "kv_b", "gate"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[kv_a, kv_b, gate],
        output_shapes=[kv_a.shape],
        output_dtypes=[mx.float32],
        grid=(kv_a.size, 1, 1),
        threadgroup=(min(kv_a.size, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── context-engineering-mlx ───────────────────────────────────────────────────
"context-engineering-mlx/src/context_engineering_mlx/infra/ops/prune.py": '''import mlx.core as mx

SCORE_MSL = """
    uint idx = thread_position_in_grid.x;
    float dim_f = (float)dim_arr[0];
    float acc = 0.0f;
    for (uint d = 0; d < dim_arr[0]; ++d)
        acc += tokens[idx * dim_arr[0] + d];
    scores[idx] = acc / dim_f;
"""

MASK_MSL = """
    uint idx = thread_position_in_grid.x;
    mask[idx] = (scores[idx] >= threshold[0]) ? (uint8_t)1 : (uint8_t)0;
"""

def score_and_prune(tokens: mx.array, threshold: float = 0.1,
                    stream: mx.Stream = mx.gpu) -> tuple[mx.array, mx.array]:
    """Returns (scores, mask) pruning low-salience tokens."""
    seq_len, dim = tokens.shape

    score_k = mx.fast.metal_kernel(
        name="ctx_score_kernel",
        input_names=["tokens", "dim_arr"],
        output_names=["scores"],
        source=SCORE_MSL,
        ensure_row_contiguous=True
    )
    scores_out = score_k(
        inputs=[tokens, mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.float32],
        grid=(seq_len, 1, 1),
        threadgroup=(min(seq_len, 256), 1, 1),
        stream=stream
    )
    scores = scores_out[0]

    mask_k = mx.fast.metal_kernel(
        name="ctx_mask_kernel",
        input_names=["scores", "threshold"],
        output_names=["mask"],
        source=MASK_MSL,
        ensure_row_contiguous=True
    )
    mask_out = mask_k(
        inputs=[scores, mx.array([threshold], dtype=mx.float32)],
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.uint8],
        grid=(seq_len, 1, 1),
        threadgroup=(min(seq_len, 256), 1, 1),
        stream=stream
    )
    return scores, mask_out[0]
''',

# ── cuda-bridge-mlx ───────────────────────────────────────────────────────────
"cuda-bridge-mlx/src/cuda_bridge_mlx/infra/ops/layout.py": '''import mlx.core as mx

MSL_SOURCE = """
    // NCHW -> NHWC layout conversion (or generic row-major reorder)
    // For 2D: this is effectively a copy (already contiguous)
    uint idx = thread_position_in_grid.x;
    out[idx] = inp[idx];
"""

def cuda_to_metal(tensor: mx.array, stream: mx.Stream = mx.gpu) -> mx.array:
    """Converts CUDA NCHW layout tensor to Metal-native NHWC (2D passthrough)."""
    kernel = mx.fast.metal_kernel(
        name="cuda_to_metal_layout",
        input_names=["inp"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[tensor],
        output_shapes=[tensor.shape],
        output_dtypes=[tensor.dtype],
        grid=(tensor.size, 1, 1),
        threadgroup=(min(tensor.size, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── eagle-mlx ─────────────────────────────────────────────────────────────────
"eagle-mlx/src/eagle_mlx/infra/ops/draft.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.x;
    uint d   = thread_position_in_grid.y;
    uint dim_val = dim_arr[0];
    float val = 0.0f;
    for (uint k = 0; k < dim_val; ++k)
        val += hidden[tok * dim_val + k] * proj_weight[k * dim_val + d];
    draft_logits[tok * dim_val + d] = val;
"""

def eagle_draft(hidden: mx.array, proj_weight: mx.array,
                stream: mx.Stream = mx.gpu) -> mx.array:
    """EAGLE speculative decoding: project hidden states to draft logits."""
    seq_len, dim = hidden.shape
    kernel = mx.fast.metal_kernel(
        name="eagle_draft_kernel",
        input_names=["hidden", "proj_weight", "dim_arr"],
        output_names=["draft_logits"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[hidden, proj_weight, mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_len, dim, 1),
        threadgroup=(1, min(dim, 64), 1),
        stream=stream
    )
    return out[0]
''',

# ── evol-kv-mlx ───────────────────────────────────────────────────────────────
"evol-kv-mlx/src/evol_kv_mlx/infra/ops/score.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint idx = thread_position_in_grid.x;
    // EvolKV: importance score = mean absolute attention weight per KV slot
    uint n_heads = n_heads_arr[0];
    float acc = 0.0f;
    for (uint h = 0; h < n_heads; ++h)
        acc += fabs(attn_weights[h * n_arr[0] + idx]);
    scores[idx] = acc / (float)n_heads;
"""

def evol_kv_score(attn_weights: mx.array,
                  stream: mx.Stream = mx.gpu) -> mx.array:
    """Computes per-KV importance scores across heads."""
    n_heads, n = attn_weights.shape
    kernel = mx.fast.metal_kernel(
        name="evol_kv_score_kernel",
        input_names=["attn_weights", "n_heads_arr", "n_arr"],
        output_names=["scores"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[attn_weights,
                mx.array([n_heads], dtype=mx.uint32),
                mx.array([n], dtype=mx.uint32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── expert-selective-mlx ──────────────────────────────────────────────────────
"expert-selective-mlx/src/expert_selective_mlx/infra/ops/gate.py": '''import mlx.core as mx

TOPK_MSL = """
    uint tok  = thread_position_in_grid.x;
    uint ne   = n_experts_arr[0];
    uint topk = top_k_arr[0];
    uint base = tok * ne;

    // Simple selection: top-k via insertion sort (small k)
    for (uint ki = 0; ki < topk; ++ki) {
        float best = -1e9f;
        uint  best_idx = 0;
        for (uint e = 0; e < ne; ++e) {
            bool already = false;
            for (uint p = 0; p < ki; ++p)
                if (indices[tok * topk + p] == e) { already = true; break; }
            if (!already && gate_logits[base + e] > best) {
                best = gate_logits[base + e];
                best_idx = e;
            }
        }
        indices[tok * topk + ki] = best_idx;
        scores[tok * topk + ki]  = best;
    }
"""

def expert_select_topk(gate_logits: mx.array, top_k: int = 2,
                        stream: mx.Stream = mx.gpu) -> tuple[mx.array, mx.array]:
    """Selects top-k experts for each token via MoE gating."""
    seq_len, n_experts = gate_logits.shape
    kernel = mx.fast.metal_kernel(
        name="expert_select_topk_kernel",
        input_names=["gate_logits", "n_experts_arr", "top_k_arr"],
        output_names=["indices", "scores"],
        source=TOPK_MSL,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[gate_logits,
                mx.array([n_experts], dtype=mx.uint32),
                mx.array([top_k], dtype=mx.uint32)],
        output_shapes=[(seq_len, top_k), (seq_len, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(seq_len, 1, 1),
        threadgroup=(1, 1, 1),
        stream=stream
    )
    return out[0], out[1]
''',

# ── flash-infer-mlx ───────────────────────────────────────────────────────────
"flash-infer-mlx/src/flash_infer_mlx/infra/ops/infer.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok  = thread_position_in_grid.x;
    uint d    = thread_position_in_grid.y;
    uint page = page_table[tok];
    uint dim  = dim_arr[0];
    uint psize= page_size_arr[0];
    // Decode: attend q[tok,d] against paged KV
    float acc = 0.0f;
    for (uint pi = 0; pi < psize; ++pi) {
        uint kv_idx = page * psize * dim + pi * dim + d;
        acc += q[tok * dim + d] * kv_pages[kv_idx];
    }
    out[tok * dim + d] = acc;
"""

def flash_infer_decode(q: mx.array, kv_pages: mx.array, page_table: mx.array,
                       page_size: int = 16,
                       stream: mx.Stream = mx.gpu) -> mx.array:
    """Paged attention decode using FlashInfer-style layout."""
    seq_len, dim = q.shape
    kernel = mx.fast.metal_kernel(
        name="flash_infer_decode_kernel",
        input_names=["q", "kv_pages", "page_table", "dim_arr", "page_size_arr"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[q, kv_pages, page_table,
                mx.array([dim], dtype=mx.uint32),
                mx.array([page_size], dtype=mx.uint32)],
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_len, dim, 1),
        threadgroup=(1, min(dim, 64), 1),
        stream=stream
    )
    return out[0]
''',

# ── forge-mlx ─────────────────────────────────────────────────────────────────
"forge-mlx/src/forge_mlx/infra/ops/forge.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint row = thread_position_in_grid.x;
    uint col = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    float val = bias[col];
    for (uint k = 0; k < dim; ++k)
        val += x[row * dim + k] * w[col * dim + k];
    out[row * out_dim_arr[0] + col] = val;
"""

def forge_layer_build(x: mx.array, w: mx.array, bias: mx.array,
                      stream: mx.Stream = mx.gpu) -> mx.array:
    """Generic fused linear + bias layer build."""
    seq_len, in_dim = x.shape
    out_dim = w.shape[0]
    kernel = mx.fast.metal_kernel(
        name="forge_layer_kernel",
        input_names=["x", "w", "bias", "dim_arr", "out_dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[x, w, bias,
                mx.array([in_dim], dtype=mx.uint32),
                mx.array([out_dim], dtype=mx.uint32)],
        output_shapes=[(seq_len, out_dim)],
        output_dtypes=[mx.float32],
        grid=(seq_len, out_dim, 1),
        threadgroup=(1, min(out_dim, 32), 1),
        stream=stream
    )
    return out[0]
''',

# ── fused-moe-mlx ─────────────────────────────────────────────────────────────
"fused-moe-mlx/src/fused_moe_mlx/infra/ops/moe.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok    = thread_position_in_grid.x;
    uint out_d  = thread_position_in_grid.y;
    uint in_dim = in_dim_arr[0];
    uint out_dm = out_dim_arr[0];
    uint topk   = topk_arr[0];
    uint ne     = n_experts_arr[0];
    // Weighted sum over top-k expert outputs
    float acc = 0.0f;
    for (uint ki = 0; ki < topk; ++ki) {
        uint eid   = expert_ids[tok * topk + ki];
        float gate = gate_weights[tok * topk + ki];
        // w1 projection
        float h = 0.0f;
        for (uint d = 0; d < in_dim; ++d)
            h += x[tok * in_dim + d] * w1[eid * in_dim * out_dm + d * out_dm + out_d];
        // simple gelu approx: h * sigmoid(1.702 * h)
        h = h / (1.0f + exp(-1.702f * h));
        // w2 projection (single output dim aggregation)
        float y = 0.0f;
        for (uint od = 0; od < out_dm; ++od)
            y += h * w2[eid * out_dm * in_dim + od * in_dim + out_d % in_dim];
        acc += gate * y;
    }
    out[tok * out_dm + out_d] = acc;
"""

def fused_moe(x: mx.array, w1: mx.array, w2: mx.array,
              expert_ids: mx.array, gate_weights: mx.array,
              stream: mx.Stream = mx.gpu) -> mx.array:
    """Fused Mixture-of-Experts forward pass."""
    seq_len, in_dim = x.shape
    n_experts, out_dim, _ = w1.shape
    topk = expert_ids.shape[1]
    kernel = mx.fast.metal_kernel(
        name="fused_moe_kernel",
        input_names=["x","w1","w2","expert_ids","gate_weights",
                     "in_dim_arr","out_dim_arr","topk_arr","n_experts_arr"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[x, w1, w2, expert_ids, gate_weights,
                mx.array([in_dim],    dtype=mx.uint32),
                mx.array([out_dim],   dtype=mx.uint32),
                mx.array([topk],      dtype=mx.uint32),
                mx.array([n_experts], dtype=mx.uint32)],
        output_shapes=[(seq_len, out_dim)],
        output_dtypes=[mx.float32],
        grid=(seq_len, out_dim, 1),
        threadgroup=(1, min(out_dim, 32), 1),
        stream=stream
    )
    return out[0]
''',

# ── h2o-mlx ───────────────────────────────────────────────────────────────────
"h2o-mlx/src/h2o_mlx/infra/ops/h2o.py": '''import mlx.core as mx

MSL_SOURCE = """
    // H2O: keep top keep_n tokens by attention sum; others evicted
    uint idx = thread_position_in_grid.x;
    uint n   = n_arr[0];
    uint kn  = keep_n_arr[0];

    float my_score = attn_sum[idx];
    uint rank = 0;
    for (uint j = 0; j < n; ++j)
        if (attn_sum[j] > my_score) rank++;
    keep[idx] = (rank < kn) ? (uint8_t)1 : (uint8_t)0;
"""

def h2o_evict(attn_sum: mx.array, keep_ratio: float = 0.5,
               stream: mx.Stream = mx.gpu) -> mx.array:
    """H2O KV eviction: returns keep mask (1=keep, 0=evict)."""
    n = attn_sum.shape[0]
    keep_n = max(1, int(n * keep_ratio))
    kernel = mx.fast.metal_kernel(
        name="h2o_evict_kernel",
        input_names=["attn_sum", "n_arr", "keep_n_arr"],
        output_names=["keep"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[attn_sum,
                mx.array([n],      dtype=mx.uint32),
                mx.array([keep_n], dtype=mx.uint32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.uint8],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── hqq-mlx ───────────────────────────────────────────────────────────────────
"hqq-mlx/src/hqq_mlx/infra/ops/hqq.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint idx       = thread_position_in_grid.x;
    uint inner_dim = inner_dim_arr[0];
    uint ch        = idx / inner_dim;
    float s = scale[ch];
    float z = zero[ch];
    out[idx] = ((float)((int)qw[idx]) - z) * s;
"""

def hqq_dequantize(qw: mx.array, scale: mx.array, zero: mx.array,
                   stream: mx.Stream = mx.gpu) -> mx.array:
    """HQQ half-quadratic quantization dequantize: float = (q - zero) * scale."""
    assert len(qw.shape) == 2
    num_ch, inner = qw.shape
    kernel = mx.fast.metal_kernel(
        name="hqq_dequant_kernel",
        input_names=["qw", "scale", "zero", "inner_dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[qw, scale, zero, mx.array([inner], dtype=mx.uint32)],
        output_shapes=[(num_ch, inner)],
        output_dtypes=[mx.float32],
        grid=(num_ch * inner, 1, 1),
        threadgroup=(min(num_ch * inner, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── inference-time-compute-mlx ────────────────────────────────────────────────
"inference-time-compute-mlx/src/inference_time_compute_mlx/infra/ops/itc.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint idx   = thread_position_in_grid.x;
    float e    = entropy[idx];
    float base = base_arr[0];
    float maxs = max_s_arr[0];
    // scale = clamp(base * exp(e), 1, max_s)
    float s = base * exp(e);
    if (s < 1.0f) s = 1.0f;
    if (s > maxs) s = maxs;
    steps[idx] = s;
"""

def itc_scale_steps(entropy: mx.array, base: float = 1.0, max_s: float = 8.0,
                    stream: mx.Stream = mx.gpu) -> mx.array:
    """Inference-Time Compute: scale reasoning steps by token entropy."""
    n = entropy.shape[0]
    kernel = mx.fast.metal_kernel(
        name="itc_scale_kernel",
        input_names=["entropy", "base_arr", "max_s_arr"],
        output_names=["steps"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[entropy,
                mx.array([base],  dtype=mx.float32),
                mx.array([max_s], dtype=mx.float32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── infini-attention-mlx ──────────────────────────────────────────────────────
"infini-attention-mlx/src/infini_attention_mlx/infra/ops/infini.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint i   = thread_position_in_grid.x;
    uint dim = dim_arr[0];
    // Update compressive memory: M = M + outer(v, sigma(k))
    // Here simplified: M[i] += v[i] * mean(k)
    float k_mean = 0.0f;
    for (uint d = 0; d < dim; ++d) k_mean += new_k[d];
    k_mean /= (float)dim;
    out_M[i] = mem_M[i] + new_v[i % dim] * k_mean;
    out_z[i] = mem_z[i] + k_mean;
"""

def infini_attention_update(mem_M: mx.array, mem_z: mx.array,
                             new_k: mx.array, new_v: mx.array,
                             stream: mx.Stream = mx.gpu) -> tuple[mx.array, mx.array]:
    """Infini-Attention: update compressive memory with new KV."""
    dim = new_k.shape[0]
    n   = mem_M.shape[0]
    kernel = mx.fast.metal_kernel(
        name="infini_attn_update_kernel",
        input_names=["mem_M", "mem_z", "new_k", "new_v", "dim_arr"],
        output_names=["out_M", "out_z"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[mem_M, mem_z, new_k, new_v, mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        stream=stream
    )
    return out[0], out[1]
''',

# ── kvcomp-mlx ────────────────────────────────────────────────────────────────
"kvcomp-mlx/src/kvcomp_mlx/infra/ops/compress.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint out_i  = thread_position_in_grid.x;
    uint d      = thread_position_in_grid.y;
    uint ratio  = ratio_arr[0];
    uint hidden = hidden_arr[0];
    // Average pool: merge `ratio` consecutive KV frames
    float acc = 0.0f;
    for (uint r = 0; r < ratio; ++r)
        acc += kv[out_i * ratio * hidden + r * hidden + d];
    out[out_i * hidden + d] = acc / (float)ratio;
"""

def kv_compress(kv: mx.array, ratio: int = 2,
                stream: mx.Stream = mx.gpu) -> mx.array:
    """KV-cache compression via average pooling by ratio."""
    assert len(kv.shape) == 2
    seq_len, hidden = kv.shape
    out_len = seq_len // ratio
    kernel = mx.fast.metal_kernel(
        name="kv_merge_kernel",
        input_names=["kv", "hidden_arr", "ratio_arr"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[kv,
                mx.array([hidden], dtype=mx.uint32),
                mx.array([ratio],  dtype=mx.uint32)],
        output_shapes=[(out_len, hidden)],
        output_dtypes=[mx.float32],
        grid=(out_len, hidden, 1),
        threadgroup=(1, min(hidden, 256), 1),
        stream=stream
    )
    return out[0]
''',

# ── kvtc-mlx ──────────────────────────────────────────────────────────────────
"kvtc-mlx/src/kvtc_mlx/infra/ops/tiered.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint idx  = thread_position_in_grid.x;
    float f   = access_freq[idx];
    float hot  = hot_thresh[0];
    float warm = warm_thresh[0];
    if      (f >= hot)  tier[idx] = 0;
    else if (f >= warm) tier[idx] = 1;
    else                tier[idx] = 2;
"""

def kvtc_route(access_freq: mx.array, hot: float = 0.7, warm: float = 0.3,
               stream: mx.Stream = mx.gpu) -> mx.array:
    """KVTC: classify each KV slot into hot/warm/cold tier."""
    n = access_freq.shape[0]
    kernel = mx.fast.metal_kernel(
        name="kvtc_route_kernel",
        input_names=["access_freq", "hot_thresh", "warm_thresh"],
        output_names=["tier"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[access_freq,
                mx.array([hot],  dtype=mx.float32),
                mx.array([warm], dtype=mx.float32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.uint8],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── layerskip-mlx ─────────────────────────────────────────────────────────────
"layerskip-mlx/src/layerskip_mlx/infra/ops/skip.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok     = thread_position_in_grid.x;
    uint hd      = hidden_dim_arr[0];
    float thresh = threshold_arr[0];
    float conf   = 0.0f;
    for (uint d = 0; d < hd; ++d)
        conf += hidden[tok * hd + d] * exit_w[d];
    do_exit[tok] = (conf >= thresh) ? (uint8_t)1 : (uint8_t)0;
"""

def layerskip_gate(hidden: mx.array, exit_w: mx.array, threshold: float = 0.5,
                   stream: mx.Stream = mx.gpu) -> mx.array:
    """LayerSkip confidence-based early exit gate."""
    seq_len, hidden_dim = hidden.shape
    kernel = mx.fast.metal_kernel(
        name="layerskip_gate_kernel",
        input_names=["hidden", "exit_w", "threshold_arr", "hidden_dim_arr"],
        output_names=["do_exit"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[hidden, exit_w,
                mx.array([threshold],   dtype=mx.float32),
                mx.array([hidden_dim],  dtype=mx.uint32)],
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.uint8],
        grid=(seq_len, 1, 1),
        threadgroup=(min(seq_len, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── liger-kernel-mlx ──────────────────────────────────────────────────────────
"liger-kernel-mlx/src/liger_kernel_mlx/infra/ops/liger.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok   = thread_position_in_grid.x;
    uint vsize = vocab_size_arr[0];
    uint base  = tok * vsize;
    uint label = labels[tok];

    float max_v = logits[base];
    for (uint v = 1; v < vsize; ++v)
        if (logits[base + v] > max_v) max_v = logits[base + v];

    float sum_exp = 0.0f;
    for (uint v = 0; v < vsize; ++v)
        sum_exp += exp(logits[base + v] - max_v);

    losses[tok] = log(sum_exp) + max_v - logits[base + label];
"""

def liger_cross_entropy(logits: mx.array, labels: mx.array,
                        stream: mx.Stream = mx.gpu) -> mx.array:
    """Liger fused cross-entropy (stable log-softmax + NLL)."""
    assert len(logits.shape) == 2
    seq_len, vocab_size = logits.shape
    assert labels.dtype == mx.uint32
    kernel = mx.fast.metal_kernel(
        name="liger_ce_kernel",
        input_names=["logits", "labels", "vocab_size_arr"],
        output_names=["losses"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[logits, labels, mx.array([vocab_size], dtype=mx.uint32)],
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.float32],
        grid=(seq_len, 1, 1),
        threadgroup=(1, 1, 1),
        stream=stream
    )
    return out[0]
''',

# ── turboquant-mlx ────────────────────────────────────────────────────────────
"turboquant-mlx/src/turboquant_mlx/infra/ops/quant.py": '''import mlx.core as mx

QUANT_SOURCE = """
    uint ch   = thread_position_in_grid.y;
    uint idx  = thread_position_in_grid.x;
    uint inner = inner_dim_arr[0];
    uint flat  = ch * inner + idx;
    float q    = (input[flat] / scale[ch]) + zero_p[ch];
    if (q < -128.0f) q = -128.0f;
    if (q >  127.0f) q =  127.0f;
    output[flat] = (int8_t)round(q);
"""

DEQUANT_SOURCE = """
    uint ch    = thread_position_in_grid.y;
    uint idx   = thread_position_in_grid.x;
    uint inner = inner_dim_arr[0];
    uint flat  = ch * inner + idx;
    output[flat] = ((float)input[flat] - zero_p[ch]) * scale[ch];
"""

def quantize(x: mx.array, scale: mx.array, zero_point: mx.array,
             stream: mx.Stream = mx.gpu) -> mx.array:
    """TurboQuant: float32 → int8 per-channel quantization."""
    assert len(x.shape) == 2
    num_ch, inner = x.shape
    kernel = mx.fast.metal_kernel(
        name="turboquant_quant_int8",
        input_names=["input", "scale", "zero_p", "inner_dim_arr"],
        output_names=["output"],
        source=QUANT_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[x, scale, zero_point, mx.array([inner], dtype=mx.uint32)],
        output_shapes=[(num_ch, inner)],
        output_dtypes=[mx.int8],
        grid=(inner, num_ch, 1),
        threadgroup=(min(inner, 256), 1, 1),
        stream=stream
    )
    return out[0]

def dequantize(x: mx.array, scale: mx.array, zero_point: mx.array,
               stream: mx.Stream = mx.gpu) -> mx.array:
    """TurboQuant: int8 → float32 per-channel dequantization."""
    assert len(x.shape) == 2
    num_ch, inner = x.shape
    kernel = mx.fast.metal_kernel(
        name="turboquant_dequant_int8",
        input_names=["input", "scale", "zero_p", "inner_dim_arr"],
        output_names=["output"],
        source=DEQUANT_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[x, scale, zero_point, mx.array([inner], dtype=mx.uint32)],
        output_shapes=[(num_ch, inner)],
        output_dtypes=[mx.float32],
        grid=(inner, num_ch, 1),
        threadgroup=(min(inner, 256), 1, 1),
        stream=stream
    )
    return out[0]
''',
}

def migrate():
    ok = 0
    skip = 0
    for rel_path, content in MIGRATIONS.items():
        full_path = METAL_ROOT / rel_path
        if not full_path.parent.exists():
            print(f"  ⚠️  SKIP (dir missing): {rel_path}")
            skip += 1
            continue
        full_path.write_text(content)
        print(f"  ✅  migrated: {rel_path}")
        ok += 1
    print(f"\nDone: {ok} migrated, {skip} skipped")

if __name__ == "__main__":
    migrate()
