import mlx.core as mx

MSL_SOURCE = """
// Radix Cache: token-level radix-trie prefix matching for KV reuse
    uint entry = gid.x;
    uint match = 0;
    for (uint t = 0; t < token_len; ++t) {
        if (tokens[t] == trie_keys[entry * token_len + t]) match++;
        else break;
    }
    scores[entry] = (float)match / (float)token_len;
"""

def radix_prefix_score(tokens: mx.array, trie_keys: mx.array,
                        stream: mx.Stream = mx.cpu) -> mx.array:
    token_len = tokens.shape[0]
    trie_size = trie_keys.shape[0]
    kernel = mx.fast.metal_kernel(
        name="radix_score_kernel",
        input_names=["tokens","trie_keys","token_len","trie_size"],
        output_names=["scores"], source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(inputs=[tokens, trie_keys,
                         mx.array(token_len,dtype=mx.uint32),
                         mx.array(trie_size,dtype=mx.uint32)], grid=(trie_size,1,1), threadgroup=(256,1,1),
                 output_shapes=[(trie_size,)], output_dtypes=[mx.float32], stream=stream)
    return out[0]
