import mlx.core as mx

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
