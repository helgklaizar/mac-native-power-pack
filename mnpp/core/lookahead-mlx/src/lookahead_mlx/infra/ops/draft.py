import mlx.core as mx

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
