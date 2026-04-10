"""
migrate_metal_ops_batch2.py
===========================
Migrates remaining 38 Metal ops files to MLX 0.31.1 body-only API.
"""
import sys
from pathlib import Path

METAL_ROOT = Path("/Users/k/Documents/PROJECTS/MVP/for-mac/modules/Metal")

MIGRATIONS = {

"lm-cache-mlx/src/lm_cache_mlx/infra/ops/cache.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi  = thread_position_in_grid.x;
    uint nc  = n_cache_arr[0];
    uint best_idx = 0;
    uint best_match = 0;
    for (uint ci = 0; ci < nc; ++ci) {
        if (cache_hashes[ci] == query_hashes[qi]) {
            best_idx = ci;
            best_match = 1;
            break;
        }
    }
    hit[qi]   = (uint8_t)best_match;
    cache_idx[qi] = best_idx;
"""

def lm_cache_lookup(query_hashes: mx.array, cache_hashes: mx.array,
                    stream: mx.Stream = mx.gpu) -> tuple:
    n_q = query_hashes.shape[0]
    n_c = cache_hashes.shape[0]
    kernel = mx.fast.metal_kernel(
        name="lm_cache_lookup_kernel",
        input_names=["query_hashes", "cache_hashes", "n_cache_arr"],
        output_names=["hit", "cache_idx"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[query_hashes, cache_hashes, mx.array([n_c], dtype=mx.uint32)],
        output_shapes=[(n_q,), (n_q,)],
        output_dtypes=[mx.uint8, mx.uint32],
        grid=(n_q, 1, 1), threadgroup=(min(n_q, 256), 1, 1), stream=stream)
    return out[0], out[1]
''',

"lookahead-mlx/src/lookahead_mlx/infra/ops/draft.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.x;
    uint ng  = ngram_size_arr[0];
    uint dim = dim_arr[0];
    // project hidden to draft via ngram_table rows
    float best = -1e9f;
    uint best_n = 0;
    for (uint n = 0; n < ng; ++n) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += hidden[tok * dim + d] * ngram_table[n * dim + d];
        if (dot > best) { best = dot; best_n = n; }
    }
    draft_ids[tok] = best_n;
    draft_scores[tok] = best;
"""

def lookahead_decode(hidden: mx.array, ngram_table: mx.array, ngram_size: int = 4,
                     stream: mx.Stream = mx.gpu) -> tuple:
    seq_len, dim = hidden.shape
    kernel = mx.fast.metal_kernel(
        name="lookahead_decode_kernel",
        input_names=["hidden", "ngram_table", "ngram_size_arr", "dim_arr"],
        output_names=["draft_ids", "draft_scores"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[hidden, ngram_table,
                mx.array([ngram_size], dtype=mx.uint32),
                mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(seq_len,), (seq_len,)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(seq_len, 1, 1), threadgroup=(min(seq_len, 256), 1, 1), stream=stream)
    return out[0], out[1]
''',

"mamba-mlx/src/mamba_mlx/infra/ops/ssm.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint d = dim_arr[0];
    // SSM step: h = A*h + B*x, y = C*h
    float h_new = A[i] * h_prev[i] + B[i] * x[i % d];
    h_out[i]    = h_new;
    y_out[i]    = C[i] * h_new;
"""

def mamba_ssm_step(x: mx.array, A: mx.array, B: mx.array, C: mx.array,
                   h_prev: mx.array, stream: mx.Stream = mx.gpu) -> tuple:
    d = A.shape[0]
    kernel = mx.fast.metal_kernel(
        name="mamba_ssm_step_kernel",
        input_names=["x", "A", "B", "C", "h_prev", "dim_arr"],
        output_names=["h_out", "y_out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[x, A, B, C, h_prev, mx.array([d], dtype=mx.uint32)],
        output_shapes=[(d,), (d,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(d, 1, 1), threadgroup=(min(d, 256), 1, 1), stream=stream)
    return out[0], out[1]
''',

"medusa-mlx/src/medusa_mlx/infra/ops/heads.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint head = thread_position_in_grid.y;
    uint d    = thread_position_in_grid.x;
    uint dim  = dim_arr[0];
    uint nh   = n_heads_arr[0];
    float val = 0.0f;
    for (uint k = 0; k < dim; ++k)
        val += hidden[k] * heads_w[head * dim * dim + d * dim + k];
    logits[head * dim + d] = val;
"""

def medusa_draft_heads(hidden: mx.array, heads_w: mx.array,
                       stream: mx.Stream = mx.gpu) -> mx.array:
    dim = hidden.shape[0]
    n_heads = heads_w.shape[0]
    kernel = mx.fast.metal_kernel(
        name="medusa_draft_heads_kernel",
        input_names=["hidden", "heads_w", "dim_arr", "n_heads_arr"],
        output_names=["logits"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[hidden, heads_w,
                mx.array([dim], dtype=mx.uint32),
                mx.array([n_heads], dtype=mx.uint32)],
        output_shapes=[(n_heads, dim)],
        output_dtypes=[mx.float32],
        grid=(dim, n_heads, 1), threadgroup=(min(dim, 32), 1, 1), stream=stream)
    return out[0]
''',

"metal-scheduler-mlx/src/metal_scheduler_mlx/infra/ops/schedule.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint n = n_ops_arr[0];
    // Ready = no unfinished deps (dep cost sum == 0)
    float dep_sum = 0.0f;
    for (uint j = 0; j < n; ++j)
        dep_sum += op_deps[i * n + j] * (float)(j > i ? 1 : 0);
    ready[i] = (dep_sum == 0.0f) ? (uint8_t)1 : (uint8_t)0;
    priority[i] = op_costs[i] * ready[i];
"""

def metal_schedule_ops(op_costs: mx.array, op_deps: mx.array,
                       stream: mx.Stream = mx.gpu) -> tuple:
    n = op_costs.shape[0]
    kernel = mx.fast.metal_kernel(
        name="metal_schedule_kernel",
        input_names=["op_costs", "op_deps", "n_ops_arr"],
        output_names=["ready", "priority"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[op_costs, op_deps, mx.array([n], dtype=mx.uint32)],
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.uint8, mx.float32],
        grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1), stream=stream)
    return out[0], out[1]
''',

"mimo-flash-mlx/src/mimo_flash_mlx/infra/ops/mimo.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint idx = thread_position_in_grid.x;
    // MIMO: combine two query streams against shared KV
    out[idx] = (q1[idx] * k[idx] + q2[idx] * k[idx]) * v[idx] * 0.5f;
"""

def mimo_flash_attention(q1: mx.array, q2: mx.array, k: mx.array, v: mx.array,
                         stream: mx.Stream = mx.gpu) -> mx.array:
    assert q1.shape == q2.shape == k.shape == v.shape
    kernel = mx.fast.metal_kernel(
        name="mimo_flash_attn_kernel",
        input_names=["q1", "q2", "k", "v"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q1, q2, k, v],
        output_shapes=[q1.shape],
        output_dtypes=[mx.float32],
        grid=(q1.size, 1, 1), threadgroup=(min(q1.size, 256), 1, 1), stream=stream)
    return out[0]
''',

"minference-mlx/src/minference_mlx/infra/ops/sparse.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi = thread_position_in_grid.x;
    uint d  = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint nk  = n_keys_arr[0];
    // Sparse attention: attend only top-k key positions (simplified: first nk)
    float acc = 0.0f;
    for (uint ki = 0; ki < nk; ++ki)
        acc += q[qi * dim + d] * k[ki * dim + d] * v[ki * dim + d];
    out[qi * dim + d] = acc;
"""

def minference_sparse_attn(q: mx.array, k: mx.array, v: mx.array,
                            n_keys: int = 16, stream: mx.Stream = mx.gpu) -> mx.array:
    seq_len, dim = q.shape
    kernel = mx.fast.metal_kernel(
        name="minference_sparse_attn_kernel",
        input_names=["q", "k", "v", "dim_arr", "n_keys_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q, k, v,
                mx.array([dim], dtype=mx.uint32),
                mx.array([min(n_keys, k.shape[0])], dtype=mx.uint32)],
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_len, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"mla-mlx/src/mla_mlx/infra/ops/compress.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.x;
    uint cd  = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint cdim = cdim_arr[0];
    float val = 0.0f;
    for (uint d = 0; d < dim; ++d)
        val += kv[tok * dim + d] * down_proj[cd * dim + d];
    kv_c[tok * cdim + cd] = val;
"""

def mla_compress_kv(kv: mx.array, down_proj: mx.array,
                    stream: mx.Stream = mx.gpu) -> mx.array:
    seq_len, dim = kv.shape
    compressed_dim = down_proj.shape[0]
    kernel = mx.fast.metal_kernel(
        name="mla_compress_kv_kernel",
        input_names=["kv", "down_proj", "dim_arr", "cdim_arr"],
        output_names=["kv_c"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[kv, down_proj,
                mx.array([dim], dtype=mx.uint32),
                mx.array([compressed_dim], dtype=mx.uint32)],
        output_shapes=[(seq_len, compressed_dim)],
        output_dtypes=[mx.float32],
        grid=(seq_len, compressed_dim, 1),
        threadgroup=(1, min(compressed_dim, 32), 1), stream=stream)
    return out[0]
''',

"multi-agent-orchestrator-mlx/src/multi_agent_orchestrator_mlx/infra/ops/route.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint ti = thread_position_in_grid.x;
    uint na = n_agents_arr[0];
    uint dim = dim_arr[0];
    float best_score = -1e9f;
    uint best_agent = 0;
    for (uint a = 0; a < na; ++a) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d)
            dot += task_embed[ti * dim + d] * agent_embeds[a * dim + d];
        if (dot > best_score) { best_score = dot; best_agent = a; }
    }
    assigned[ti] = best_agent;
    scores[ti]   = best_score;
"""

def agent_route(task_embed: mx.array, agent_embeds: mx.array,
                stream: mx.Stream = mx.gpu) -> tuple:
    n_tasks, dim = task_embed.shape
    n_agents = agent_embeds.shape[0]
    kernel = mx.fast.metal_kernel(
        name="agent_route_kernel",
        input_names=["task_embed", "agent_embeds", "n_agents_arr", "dim_arr"],
        output_names=["assigned", "scores"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[task_embed, agent_embeds,
                mx.array([n_agents], dtype=mx.uint32),
                mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(n_tasks,), (n_tasks,)],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(n_tasks, 1, 1), threadgroup=(min(n_tasks, 256), 1, 1), stream=stream)
    return out[0], out[1]
''',

"mxfp4-mlx/src/mxfp4_mlx/infra/ops/dequant.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint gi  = thread_position_in_grid.x;
    uint gs  = group_size_arr[0];
    uint base = gi * gs;
    float sc = scales[gi];
    for (uint i = 0; i < gs; ++i) {
        uint8_t p    = packed[gi * ((gs + 1) / 2) + i / 2];
        int nibble   = (i % 2 == 0) ? (int)(p & 0x0F) : (int)((p >> 4) & 0x0F);
        int signed_n = nibble - 8;
        out[base + i] = (float)signed_n * sc;
    }
"""

def mxfp4_dequantize(packed: mx.array, scales: mx.array, group_size: int = 32,
                     stream: mx.Stream = mx.gpu) -> mx.array:
    n_groups = scales.shape[0]
    kernel = mx.fast.metal_kernel(
        name="mxfp4_dequant_kernel",
        input_names=["packed", "scales", "group_size_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[packed, scales, mx.array([group_size], dtype=mx.uint32)],
        output_shapes=[(n_groups * group_size,)],
        output_dtypes=[mx.float32],
        grid=(n_groups, 1, 1), threadgroup=(1, 1, 1), stream=stream)
    return out[0]
''',

"nsa-attention-mlx/src/nsa_attention_mlx/infra/ops/nsa.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi  = thread_position_in_grid.x;
    uint d   = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint nk  = n_keys_arr[0];
    float acc = 0.0f;
    for (uint ki = 0; ki < nk; ++ki) {
        if (mask[qi * nk + ki]) {
            float dot = q[qi * dim + d] * k[ki * dim + d];
            acc += dot * v[ki * dim + d];
        }
    }
    out[qi * dim + d] = acc;
"""

def nsa_attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array,
                  stream: mx.Stream = mx.gpu) -> mx.array:
    seq_q, dim = q.shape
    n_keys = k.shape[0]
    kernel = mx.fast.metal_kernel(
        name="nsa_attention_kernel",
        input_names=["q", "k", "v", "mask", "dim_arr", "n_keys_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q, k, v, mask,
                mx.array([dim], dtype=mx.uint32),
                mx.array([n_keys], dtype=mx.uint32)],
        output_shapes=[(seq_q, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_q, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"omniquant-mlx/src/omniquant_mlx/infra/ops/dequant.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint idx = thread_position_in_grid.x;
    uint ch  = idx / inner_dim_arr[0];
    out[idx] = (float)qw[idx] * alpha[ch] + beta[ch];
"""

def omniquant_dequant(qw: mx.array, alpha: mx.array,
                      beta: mx.array, stream: mx.Stream = mx.gpu) -> mx.array:
    assert len(qw.shape) == 2
    num_ch, inner = qw.shape
    kernel = mx.fast.metal_kernel(
        name="omniquant_dequant_kernel",
        input_names=["qw", "alpha", "beta", "inner_dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[qw, alpha, beta, mx.array([inner], dtype=mx.uint32)],
        output_shapes=[(num_ch, inner)],
        output_dtypes=[mx.float32],
        grid=(num_ch * inner, 1, 1),
        threadgroup=(min(num_ch * inner, 256), 1, 1), stream=stream)
    return out[0]
''',

"paged-attention-mlx/src/paged_attention_mlx/infra/ops/paged.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi   = thread_position_in_grid.x;
    uint d    = thread_position_in_grid.y;
    uint dim  = dim_arr[0];
    uint bs   = block_size_arr[0];
    uint nb   = n_blocks_arr[0];
    float acc = 0.0f;
    for (uint bi = 0; bi < nb; ++bi) {
        uint blk = block_table[bi];
        for (uint t = 0; t < bs; ++t) {
            uint ki = blk * bs + t;
            acc += q[qi * dim + d] * kv_pool[ki * dim + d];
        }
    }
    out[qi * dim + d] = acc;
"""

def paged_attention(q: mx.array, kv_pool: mx.array, block_table: mx.array,
                    block_size: int = 16, stream: mx.Stream = mx.gpu) -> mx.array:
    seq_q, dim = q.shape
    n_blocks = block_table.shape[0]
    kernel = mx.fast.metal_kernel(
        name="paged_attention_kernel",
        input_names=["q", "kv_pool", "block_table", "dim_arr", "block_size_arr", "n_blocks_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q, kv_pool, block_table,
                mx.array([dim], dtype=mx.uint32),
                mx.array([block_size], dtype=mx.uint32),
                mx.array([n_blocks], dtype=mx.uint32)],
        output_shapes=[(seq_q, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_q, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"plan-and-explore-mlx/src/plan_and_explore_mlx/infra/ops/ucb.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint i    = thread_position_in_grid.x;
    float q   = q_values[i];
    float n   = (float)visit_cnt[i];
    float tv  = total_v_arr[0];
    float c   = c_puct_arr[0];
    ucb_score[i] = q + c * sqrt(log(tv) / (1.0f + n));
"""

def plan_explore_score(q_values: mx.array, visit_cnt: mx.array,
                       total_visits: float, c_puct: float = 1.41,
                       stream: mx.Stream = mx.gpu) -> mx.array:
    n = q_values.shape[0]
    kernel = mx.fast.metal_kernel(
        name="plan_explore_kernel",
        input_names=["q_values", "visit_cnt", "total_v_arr", "c_puct_arr"],
        output_names=["ucb_score"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q_values, visit_cnt,
                mx.array([total_visits], dtype=mx.float32),
                mx.array([c_puct], dtype=mx.float32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1), stream=stream)
    return out[0]
''',

"polarquant-mlx/src/polarquant_mlx/infra/ops/rotate.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    uint in_d  = in_dim_arr[0];
    uint out_d = out_dim_arr[0];
    float s = 0.0f;
    for (uint d = 0; d < in_d; ++d)
        s += w[row * in_d + d] * rot_mat[col * in_d + d];
    out[row * out_d + col] = s;
"""

def polarquant_rotate(w: mx.array, rot_mat: mx.array,
                      stream: mx.Stream = mx.gpu) -> mx.array:
    rows, in_dim = w.shape
    out_dim = rot_mat.shape[0]
    kernel = mx.fast.metal_kernel(
        name="polar_rotate_kernel",
        input_names=["w", "rot_mat", "in_dim_arr", "out_dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[w, rot_mat,
                mx.array([in_dim], dtype=mx.uint32),
                mx.array([out_dim], dtype=mx.uint32)],
        output_shapes=[(rows, out_dim)],
        output_dtypes=[mx.float32],
        grid=(out_dim, rows, 1), threadgroup=(min(out_dim, 32), 1, 1), stream=stream)
    return out[0]
''',

"pyramid-kv-mlx/src/pyramid_kv_mlx/infra/ops/pool.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint blk = thread_position_in_grid.x;
    uint d   = thread_position_in_grid.y;
    uint ws  = win_size_arr[0];
    uint hid = hidden_arr[0];
    uint sl  = seq_len_arr[0];
    float s = 0.0f; uint cnt = 0;
    for (uint t = blk * ws; t < ws * (blk + 1) && t < sl; ++t) {
        s += kv[t * hid + d]; cnt++;
    }
    out[blk * hid + d] = (cnt > 0) ? s / (float)cnt : 0.0f;
"""

def pyramid_kv_pool(kv: mx.array, win_size: int = 4,
                    stream: mx.Stream = mx.gpu) -> mx.array:
    seq_len, hidden = kv.shape
    out_len = (seq_len + win_size - 1) // win_size
    kernel = mx.fast.metal_kernel(
        name="pyramid_pool_kernel",
        input_names=["kv", "win_size_arr", "hidden_arr", "seq_len_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[kv,
                mx.array([win_size], dtype=mx.uint32),
                mx.array([hidden], dtype=mx.uint32),
                mx.array([seq_len], dtype=mx.uint32)],
        output_shapes=[(out_len, hidden)],
        output_dtypes=[mx.float32],
        grid=(out_len, hidden, 1), threadgroup=(1, min(hidden, 256), 1), stream=stream)
    return out[0]
''',

"qaq-mlx/src/qaq_mlx/infra/ops/quant.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.y;
    uint d   = thread_position_in_grid.x;
    uint id  = inner_dim_arr[0];
    uint b   = bits[tok];
    float levels = (float)((1 << (b - 1)) - 1);
    float max_v = 0.0f;
    for (uint i = 0; i < id; ++i)
        if (fabs(input[tok * id + i]) > max_v) max_v = fabs(input[tok * id + i]);
    float scale = (max_v > 0.0f) ? levels / max_v : 1.0f;
    scales[tok] = 1.0f / scale;
    float v = input[tok * id + d] * scale;
    if (v < -levels) v = -levels;
    if (v >  levels) v =  levels;
    out[tok * id + d] = (int8_t)round(v);
"""

def qaq_adaptive_quant(x: mx.array, bits: mx.array,
                       stream: mx.Stream = mx.gpu) -> tuple:
    seq_len, inner_dim = x.shape
    kernel = mx.fast.metal_kernel(
        name="qaq_quant_kernel",
        input_names=["input", "bits", "inner_dim_arr"],
        output_names=["out", "scales"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[x, bits, mx.array([inner_dim], dtype=mx.uint32)],
        output_shapes=[(seq_len, inner_dim), (seq_len,)],
        output_dtypes=[mx.int8, mx.float32],
        grid=(inner_dim, seq_len, 1),
        threadgroup=(min(inner_dim, 256), 1, 1), stream=stream)
    return out[0], out[1]
''',

"qjl-quant-mlx/src/qjl_quant_mlx/infra/ops/dequant.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.y;
    uint o   = thread_position_in_grid.x;
    uint pd  = proj_dim_arr[0];
    uint od  = out_dim_arr[0];
    float s  = 0.0f;
    for (uint p = 0; p < pd; ++p)
        s += (float)codes[tok * pd + p] * scales[p] * proj[o * pd + p];
    out[tok * od + o] = s;
"""

def qjl_dequantize(codes: mx.array, scales: mx.array, proj: mx.array,
                   stream: mx.Stream = mx.gpu) -> mx.array:
    assert codes.dtype == mx.int8
    seq_len, proj_dim = codes.shape
    out_dim = proj.shape[0]
    kernel = mx.fast.metal_kernel(
        name="qjl_dequant_kernel",
        input_names=["codes", "scales", "proj", "proj_dim_arr", "out_dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[codes, scales, proj,
                mx.array([proj_dim], dtype=mx.uint32),
                mx.array([out_dim], dtype=mx.uint32)],
        output_shapes=[(seq_len, out_dim)],
        output_dtypes=[mx.float32],
        grid=(out_dim, seq_len, 1),
        threadgroup=(min(out_dim, 32), 1, 1), stream=stream)
    return out[0]
''',

"quarot-mlx/src/quarot_mlx/infra/ops/hadamard.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.y;
    uint i   = thread_position_in_grid.x;
    uint n   = n_arr[0];
    float s  = 0.0f;
    float inv_sqrt_n = 1.0f / sqrt((float)n);
    for (uint j = 0; j < n; ++j) {
        uint pc = (uint)popcount(i & j);
        float h = (pc % 2 == 0) ? 1.0f : -1.0f;
        s += h * x[tok * n + j];
    }
    out[tok * n + i] = s * inv_sqrt_n;
"""

def quarot_hadamard_transform(x: mx.array, stream: mx.Stream = mx.gpu) -> mx.array:
    if len(x.shape) == 1:
        x = x.reshape([1, -1])
        squeeze = True
    else:
        squeeze = False
    seq_len, n = x.shape
    kernel = mx.fast.metal_kernel(
        name="hadamard_transform_kernel",
        input_names=["x", "n_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[x, mx.array([n], dtype=mx.uint32)],
        output_shapes=[(seq_len, n)],
        output_dtypes=[mx.float32],
        grid=(n, seq_len, 1), threadgroup=(min(n, 32), 1, 1), stream=stream)
    result = out[0]
    return result.reshape([-1]) if squeeze else result
''',

"radix-cache-mlx/src/radix_cache_mlx/infra/ops/score.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint ti = thread_position_in_grid.x;
    uint nk = n_keys_arr[0];
    uint nl = tok_len_arr[0];
    uint best_len = 0;
    for (uint ki = 0; ki < nk; ++ki) {
        uint match = 0;
        for (uint l = 0; l < nl; ++l) {
            if (tokens[ti * nl + l] == trie_keys[ki * nl + l]) match++;
            else break;
        }
        if (match > best_len) best_len = match;
    }
    prefix_len[ti] = best_len;
"""

def radix_prefix_score(tokens: mx.array, trie_keys: mx.array,
                       stream: mx.Stream = mx.gpu) -> mx.array:
    n_toks, tok_len = tokens.shape
    n_keys = trie_keys.shape[0]
    kernel = mx.fast.metal_kernel(
        name="radix_prefix_score_kernel",
        input_names=["tokens", "trie_keys", "n_keys_arr", "tok_len_arr"],
        output_names=["prefix_len"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[tokens, trie_keys,
                mx.array([n_keys], dtype=mx.uint32),
                mx.array([tok_len], dtype=mx.uint32)],
        output_shapes=[(n_toks,)],
        output_dtypes=[mx.uint32],
        grid=(n_toks, 1, 1), threadgroup=(min(n_toks, 256), 1, 1), stream=stream)
    return out[0]
''',

"ring-attention-mlx/src/ring_attention_mlx/infra/ops/ring.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi = thread_position_in_grid.x;
    uint d  = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint nk  = n_keys_arr[0];
    float acc = 0.0f;
    for (uint ki = 0; ki < nk; ++ki)
        acc += q[qi * dim + d] * k[ki * dim + d] * v[ki * dim + d];
    out[qi * dim + d] += acc;
"""

def ring_attention_step(q: mx.array, k: mx.array, v: mx.array,
                        acc: mx.array, stream: mx.Stream = mx.gpu) -> mx.array:
    seq_q, dim = q.shape
    n_keys = k.shape[0]
    kernel = mx.fast.metal_kernel(
        name="ring_attention_step_kernel",
        input_names=["q", "k", "v", "out", "dim_arr", "n_keys_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    result = kernel(
        inputs=[q, k, v, acc,
                mx.array([dim], dtype=mx.uint32),
                mx.array([n_keys], dtype=mx.uint32)],
        output_shapes=[(seq_q, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_q, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return result[0]
''',

"rocket-kv-mlx/src/rocket_kv_mlx/infra/ops/evict.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint i   = thread_position_in_grid.x;
    float a  = alpha_arr[0];
    float sc = importance[i] * a + recency[i] * (1.0f - a);
    keep_mask[i] = (sc >= threshold[0]) ? (uint8_t)1 : (uint8_t)0;
"""

def rocket_kv_evict(importance: mx.array, recency: mx.array, alpha: float = 0.6,
                    threshold: float = 0.3, stream: mx.Stream = mx.gpu) -> mx.array:
    n = importance.shape[0]
    kernel = mx.fast.metal_kernel(
        name="rocket_kv_evict_kernel",
        input_names=["importance", "recency", "alpha_arr", "threshold"],
        output_names=["keep_mask"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[importance, recency,
                mx.array([alpha], dtype=mx.float32),
                mx.array([threshold], dtype=mx.float32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.uint8],
        grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1), stream=stream)
    return out[0]
''',

"sage-attention-mlx/src/sage_attention_mlx/infra/ops/sage.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi = thread_position_in_grid.x;
    uint d  = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint nk  = n_keys_arr[0];
    // SAGE: smooth k before attention
    float acc = 0.0f;
    for (uint ki = 0; ki < nk; ++ki) {
        float k_smooth = k[ki * dim + d] - smooth_k[d];
        acc += q[qi * dim + d] * k_smooth * v[ki * dim + d];
    }
    out[qi * dim + d] = acc;
"""

def sage_attention(q: mx.array, k: mx.array, v: mx.array, smooth_k: mx.array,
                   stream: mx.Stream = mx.gpu) -> mx.array:
    seq_q, dim = q.shape
    n_keys = k.shape[0]
    kernel = mx.fast.metal_kernel(
        name="sage_attn_kernel",
        input_names=["q", "k", "v", "smooth_k", "dim_arr", "n_keys_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q, k, v, smooth_k,
                mx.array([dim], dtype=mx.uint32),
                mx.array([n_keys], dtype=mx.uint32)],
        output_shapes=[(seq_q, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_q, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"search-decode-mlx/src/search_decode_mlx/infra/ops/beam.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint b   = thread_position_in_grid.x;
    uint v   = thread_position_in_grid.y;
    uint vs  = vocab_size_arr[0];
    new_scores[b * vs + v] = beam_scores[b] + log_probs[b * vs + v];
"""

def beam_search_step(log_probs: mx.array, beam_scores: mx.array,
                     stream: mx.Stream = mx.gpu) -> mx.array:
    n_beams, vocab_size = log_probs.shape
    kernel = mx.fast.metal_kernel(
        name="beam_search_step_kernel",
        input_names=["log_probs", "beam_scores", "vocab_size_arr"],
        output_names=["new_scores"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[log_probs, beam_scores,
                mx.array([vocab_size], dtype=mx.uint32)],
        output_shapes=[(n_beams, vocab_size)],
        output_dtypes=[mx.float32],
        grid=(n_beams, vocab_size, 1),
        threadgroup=(1, min(vocab_size, 64), 1), stream=stream)
    return out[0]
''',

"self-spec-mlx/src/self_spec_mlx/infra/ops/draft.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.x;
    uint d   = thread_position_in_grid.y;
    uint vs  = vocab_size_arr[0];
    uint dim = dim_arr[0];
    float val = 0.0f;
    for (uint k = 0; k < dim; ++k)
        val += hidden[tok * dim + k] * lm_head[d * dim + k];
    draft_logits[tok * vs + d] = val;
"""

def self_speculative_draft(hidden: mx.array, lm_head: mx.array,
                           stream: mx.Stream = mx.gpu) -> mx.array:
    seq_len, dim = hidden.shape
    vocab_size = lm_head.shape[0]
    kernel = mx.fast.metal_kernel(
        name="self_spec_draft_kernel",
        input_names=["hidden", "lm_head", "vocab_size_arr", "dim_arr"],
        output_names=["draft_logits"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[hidden, lm_head,
                mx.array([vocab_size], dtype=mx.uint32),
                mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(seq_len, vocab_size)],
        output_dtypes=[mx.float32],
        grid=(seq_len, vocab_size, 1),
        threadgroup=(1, min(vocab_size, 32), 1), stream=stream)
    return out[0]
''',

"speculative-decode-mlx/src/speculative_decode_mlx/infra/ops/verify.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.x;
    uint vs  = vocab_size_arr[0];
    uint base = tok * vs;
    // Accept if argmax(draft) == argmax(target)
    uint draft_top = 0; float draft_max = draft_logits[base];
    uint target_top = 0; float target_max = target_logits[base];
    for (uint v = 1; v < vs; ++v) {
        if (draft_logits[base+v]  > draft_max)  { draft_max  = draft_logits[base+v];  draft_top  = v; }
        if (target_logits[base+v] > target_max) { target_max = target_logits[base+v]; target_top = v; }
    }
    accept[tok] = (draft_top == target_top) ? (uint8_t)1 : (uint8_t)0;
"""

def speculative_verify(draft_logits: mx.array, target_logits: mx.array,
                       stream: mx.Stream = mx.gpu) -> mx.array:
    seq_len, vocab_size = draft_logits.shape
    kernel = mx.fast.metal_kernel(
        name="speculative_verify_kernel",
        input_names=["draft_logits", "target_logits", "vocab_size_arr"],
        output_names=["accept"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[draft_logits, target_logits,
                mx.array([vocab_size], dtype=mx.uint32)],
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.uint8],
        grid=(seq_len, 1, 1), threadgroup=(1, 1, 1), stream=stream)
    return out[0]
''',

"spinquant-mlx/src/spinquant_mlx/infra/ops/rotate.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    uint dim = dim_arr[0];
    float val = 0.0f;
    for (uint d = 0; d < dim; ++d)
        val += w[row * dim + d] * R[col * dim + d];
    val *= scale[col];
    out[row * dim + col] = val;
"""

def spinquant_rotate_quantize(w: mx.array, R: mx.array, scale: mx.array,
                               stream: mx.Stream = mx.gpu) -> mx.array:
    rows, dim = w.shape
    kernel = mx.fast.metal_kernel(
        name="spinquant_rotate_kernel",
        input_names=["w", "R", "scale", "dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[w, R, scale, mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(rows, dim)],
        output_dtypes=[mx.float32],
        grid=(dim, rows, 1), threadgroup=(min(dim, 32), 1, 1), stream=stream)
    return out[0]
''',

"spotlight-attention-mlx/src/spotlight_attention_mlx/infra/ops/spotlight.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi = thread_position_in_grid.x;
    uint d  = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint nk  = n_keys_arr[0];
    float acc = 0.0f;
    for (uint ki = 0; ki < nk; ++ki) {
        float dot = 0.0f;
        for (uint dd = 0; dd < dim; ++dd)
            dot += q[qi * dim + dd] * k[ki * dim + dd];
        float score = dot * spotlight_scale[0];
        acc += score * v[ki * dim + d];
    }
    out[qi * dim + d] = acc;
"""

def spotlight_attention(q: mx.array, k: mx.array, v: mx.array,
                        spotlight_scale: float = 1.0,
                        stream: mx.Stream = mx.gpu) -> mx.array:
    seq_q, dim = q.shape
    n_keys = k.shape[0]
    kernel = mx.fast.metal_kernel(
        name="spotlight_attn_kernel",
        input_names=["q", "k", "v", "spotlight_scale", "dim_arr", "n_keys_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q, k, v,
                mx.array([spotlight_scale], dtype=mx.float32),
                mx.array([dim], dtype=mx.uint32),
                mx.array([n_keys], dtype=mx.uint32)],
        output_shapes=[(seq_q, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_q, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"star-attention-mlx/src/star_attention_mlx/infra/ops/star.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi = thread_position_in_grid.x;
    uint d  = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint nk  = n_keys_arr[0];
    // Star: anchor query attends all; context queries attend anchor + local
    float acc = 0.0f;
    for (uint ki = 0; ki < nk; ++ki)
        acc += q_anchor[qi * dim + d] * k[ki * dim + d] * v[ki * dim + d];
    // Add context bias
    acc += q_ctx[qi * dim + d] * 0.1f;
    out[qi * dim + d] = acc;
"""

def star_attention(q_anchor: mx.array, q_ctx: mx.array, k: mx.array, v: mx.array,
                   stream: mx.Stream = mx.gpu) -> mx.array:
    seq_q, dim = q_anchor.shape
    n_keys = k.shape[0]
    kernel = mx.fast.metal_kernel(
        name="star_attention_kernel",
        input_names=["q_anchor", "q_ctx", "k", "v", "dim_arr", "n_keys_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q_anchor, q_ctx, k, v,
                mx.array([dim], dtype=mx.uint32),
                mx.array([n_keys], dtype=mx.uint32)],
        output_shapes=[(seq_q, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_q, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"streaming-mlx/src/streaming_mlx/infra/ops/stream.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi = thread_position_in_grid.x;
    uint d  = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint wl  = win_len_arr[0];
    float acc = 0.0f;
    for (uint ki = 0; ki < wl; ++ki)
        acc += q[qi * dim + d] * k_window[ki * dim + d] * v_window[ki * dim + d];
    out[qi * dim + d] = acc;
"""

def streaming_attn_step(q: mx.array, k_window: mx.array, v_window: mx.array,
                        stream: mx.Stream = mx.gpu) -> mx.array:
    seq_q, dim = q.shape
    win_len = k_window.shape[0]
    kernel = mx.fast.metal_kernel(
        name="streaming_attn_kernel",
        input_names=["q", "k_window", "v_window", "dim_arr", "win_len_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q, k_window, v_window,
                mx.array([dim], dtype=mx.uint32),
                mx.array([win_len], dtype=mx.uint32)],
        output_shapes=[(seq_q, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_q, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"super-weight-mlx/src/super_weight_mlx/infra/ops/scale.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint idx   = thread_position_in_grid.x;
    float val  = w[idx];
    float thr  = threshold[0];
    // Super-weight: amplify weights beyond threshold
    scaled[idx] = (fabs(val) >= thr) ? val * 2.0f : val;
"""

def super_weight_scale(w: mx.array, threshold: float = 6.0,
                       stream: mx.Stream = mx.gpu) -> mx.array:
    kernel = mx.fast.metal_kernel(
        name="super_weight_scale_kernel",
        input_names=["w", "threshold"],
        output_names=["scaled"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[w, mx.array([threshold], dtype=mx.float32)],
        output_shapes=[w.shape],
        output_dtypes=[mx.float32],
        grid=(w.size, 1, 1), threadgroup=(min(w.size, 256), 1, 1), stream=stream)
    return out[0]
''',

"tiered-cache-mlx/src/tiered_cache_mlx/infra/ops/promote.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint i    = thread_position_in_grid.x;
    float s   = scores[i];
    float h   = hot_arr[0];
    float e   = evict_arr[0];
    if      (s >= h) tier[i] = 0;
    else if (s >= e) tier[i] = 1;
    else             tier[i] = 2;
"""

def tiered_cache_promote(scores: mx.array, hot: float = 0.7, evict: float = 0.2,
                          stream: mx.Stream = mx.gpu) -> mx.array:
    n = scores.shape[0]
    kernel = mx.fast.metal_kernel(
        name="tiered_cache_promote_kernel",
        input_names=["scores", "hot_arr", "evict_arr"],
        output_names=["tier"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[scores,
                mx.array([hot], dtype=mx.float32),
                mx.array([evict], dtype=mx.float32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.uint8],
        grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1), stream=stream)
    return out[0]
''',

"titans-mlx/src/titans_mlx/infra/ops/memory.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint i   = thread_position_in_grid.x;
    uint dim = dim_arr[0];
    // Titans: neural long-term memory update
    // W_mem <- W_mem - lr * (W_mem * x - x) (simplified gradient step)
    float xi = x[i % dim];
    float lr = lr_arr[0];
    float pred = 0.0f;
    for (uint d = 0; d < dim; ++d)
        pred += W_mem[i / dim * dim + d] * x[d];
    W_out[i] = W_mem[i] - lr * (pred - xi) * x[i % dim];
"""

def titans_neural_memory_step(x: mx.array, W_mem: mx.array,
                               lr: float = 0.01,
                               stream: mx.Stream = mx.gpu) -> mx.array:
    dim = x.shape[0]
    n   = W_mem.shape[0]
    kernel = mx.fast.metal_kernel(
        name="titans_memory_kernel",
        input_names=["x", "W_mem", "lr_arr", "dim_arr"],
        output_names=["W_out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[x, W_mem,
                mx.array([lr], dtype=mx.float32),
                mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1), stream=stream)
    return out[0]
''',

"tome-mlx/src/tome_mlx/infra/ops/merge.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint ti  = thread_position_in_grid.x;
    uint d   = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint n   = n_tokens_arr[0];
    uint keep = keep_arr[0];
    // ToMe: merge similar tokens — keep first `keep` as-is, average rest
    if (ti < keep) {
        merged[ti * dim + d] = tokens[ti * dim + d];
    } else {
        float acc = 0.0f; uint cnt = 0;
        for (uint j = keep; j < n; ++j) {
            if (similarity[ti * n + j] > 0.5f) {
                acc += tokens[j * dim + d]; cnt++;
            }
        }
        merged[ti * dim + d] = (cnt > 0) ? acc / (float)cnt : tokens[ti * dim + d];
    }
"""

def tome_merge_tokens(tokens: mx.array, similarity: mx.array, keep: int,
                      stream: mx.Stream = mx.gpu) -> mx.array:
    n, dim = tokens.shape
    kernel = mx.fast.metal_kernel(
        name="tome_merge_kernel",
        input_names=["tokens", "similarity", "keep_arr", "n_tokens_arr", "dim_arr"],
        output_names=["merged"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[tokens, similarity,
                mx.array([keep], dtype=mx.uint32),
                mx.array([n], dtype=mx.uint32),
                mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(keep, dim)],
        output_dtypes=[mx.float32],
        grid=(keep, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"tri-attention-mlx/src/tri_attention_mlx/infra/ops/tri.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint qi = thread_position_in_grid.x;
    uint d  = thread_position_in_grid.y;
    uint dim = dim_arr[0];
    uint nk  = n_keys_arr[0];
    float acc = 0.0f;
    for (uint ki = 0; ki < nk; ++ki)
        acc += q[qi * dim + d] * k[ki * dim + d] * (v[ki * dim + d] + v2[ki * dim + d]);
    out[qi * dim + d] = acc;
"""

def tri_attention(q: mx.array, k: mx.array, v: mx.array, v2: mx.array,
                  stream: mx.Stream = mx.gpu) -> mx.array:
    seq_q, dim = q.shape
    n_keys = k.shape[0]
    kernel = mx.fast.metal_kernel(
        name="tri_attention_kernel",
        input_names=["q", "k", "v", "v2", "dim_arr", "n_keys_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[q, k, v, v2,
                mx.array([dim], dtype=mx.uint32),
                mx.array([n_keys], dtype=mx.uint32)],
        output_shapes=[(seq_q, dim)],
        output_dtypes=[mx.float32],
        grid=(seq_q, dim, 1), threadgroup=(1, min(dim, 64), 1), stream=stream)
    return out[0]
''',

"triton-bridge-mlx/src/triton_bridge_mlx/infra/ops/bridge.py": '''import mlx.core as mx

MSL_SOURCE = """
    // Triton -> Metal layout bridge: reinterpret row-major blocks
    uint idx = thread_position_in_grid.x;
    uint M_v = M_arr[0];
    uint N_v = N_arr[0];
    uint row = idx / N_v;
    uint col = idx % N_v;
    out[row * N_v + col] = triton_buf[col * M_v + row];
"""

def triton_to_metal(triton_buf: mx.array, M: int, N: int,
                    stream: mx.Stream = mx.gpu) -> mx.array:
    kernel = mx.fast.metal_kernel(
        name="triton_bridge_kernel",
        input_names=["triton_buf", "M_arr", "N_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[triton_buf,
                mx.array([M], dtype=mx.uint32),
                mx.array([N], dtype=mx.uint32)],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
        grid=(M * N, 1, 1), threadgroup=(min(M * N, 256), 1, 1), stream=stream)
    return out[0]
''',

"verifiable-rewards-mlx/src/verifiable_rewards_mlx/infra/ops/reward.py": '''import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.x;
    uint n   = n_tokens_arr[0];
    float correct = (float)correct_tokens[tok];
    float logp    = pred_logprobs[tok];
    // Reward: log-prob of correct token normalized
    reward[tok] = logp * correct;
"""

def compute_verifiable_reward(pred_logprobs: mx.array, correct_tokens: mx.array,
                               stream: mx.Stream = mx.gpu) -> mx.array:
    n = pred_logprobs.shape[0]
    kernel = mx.fast.metal_kernel(
        name="verifiable_reward_kernel",
        input_names=["pred_logprobs", "correct_tokens", "n_tokens_arr"],
        output_names=["reward"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[pred_logprobs, correct_tokens,
                mx.array([n], dtype=mx.uint32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1), stream=stream)
    return out[0]
''',
}

def find_ops_file(mod_path: Path) -> Path | None:
    for f in mod_path.rglob("*.py"):
        if "/ops/" in str(f) and "__init__" not in f.name:
            return f
    return None

def migrate():
    ok = skip = notfound = 0
    for rel_path, content in MIGRATIONS.items():
        full_path = METAL_ROOT / rel_path
        if not full_path.parent.exists():
            # Try to find the actual ops file
            mod_name = rel_path.split("/")[0]
            mod_path = METAL_ROOT / mod_name
            actual = find_ops_file(mod_path)
            if actual:
                actual.write_text(content)
                print(f"  ✅  migrated (auto-path): {actual.relative_to(METAL_ROOT)}")
                ok += 1
            else:
                print(f"  ❌  NOT FOUND: {mod_name}")
                notfound += 1
            continue
        full_path.write_text(content)
        print(f"  ✅  migrated: {rel_path.split('/')[-1]} ({rel_path.split('/')[0]})")
        ok += 1
    print(f"\nDone: {ok} migrated, {skip} skipped, {notfound} not found")

if __name__ == "__main__":
    migrate()
