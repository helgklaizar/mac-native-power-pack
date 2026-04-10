import mlx.core as mx

MSL_SOURCE = """
// Lookahead Decoding: parallel n-gram draft generation
    uint step = gid.x;
    float best_score = -1e9f;
    uint best_tok = 0;
    for (uint v = 0; v < ngram_vocab; ++v) {
        float s = 0.0f;
        for (uint d = 0; d < hidden_dim; ++d)
            s += hidden[step * hidden_dim + d] * ngram_table[v * hidden_dim + d];
        if (s > best_score) { best_score = s; best_tok = v; }
    }
    draft_tokens[step] = best_tok;
"""

def lookahead_decode(hidden: mx.array, ngram_table: mx.array, ngram_size: int = 4,
                     stream: mx.Stream = mx.cpu) -> mx.array:
    steps, hidden_dim = hidden.shape
    ngram_vocab = ngram_table.shape[0]
    kernel = mx.fast.metal_kernel(
        name="lookahead_draft_kernel",
        input_names=["hidden","ngram_table","hidden_dim","ngram_size","ngram_vocab"],
        output_names=["draft_tokens"], source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(inputs=[hidden, ngram_table,
                         mx.array(hidden_dim,dtype=mx.uint32),
                         mx.array(ngram_size,dtype=mx.uint32),
                         mx.array(ngram_vocab,dtype=mx.uint32)], grid=(steps,1,1), threadgroup=(1,1,1),
                 output_shapes=[(steps,)], output_dtypes=[mx.uint32], stream=stream)
    return out[0]
