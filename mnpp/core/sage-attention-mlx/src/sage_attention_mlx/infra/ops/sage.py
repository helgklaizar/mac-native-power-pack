import mlx.core as mx

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
