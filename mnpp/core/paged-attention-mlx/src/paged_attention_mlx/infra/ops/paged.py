import mlx.core as mx

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
