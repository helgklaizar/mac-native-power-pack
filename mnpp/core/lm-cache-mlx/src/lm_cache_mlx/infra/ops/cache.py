import mlx.core as mx

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
