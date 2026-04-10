import mlx.core as mx

MSL_SOURCE = """
// Medusa: multiple parallel draft heads projection
    uint head = gid.z; uint tok = gid.y; uint v = gid.x;
    if (v >= vocab_size || head >= num_heads) return;
    float s = 0.0f;
    for (uint d = 0; d < hidden_dim; ++d)
        s += hidden[tok * hidden_dim + d] *
             heads_w[head * vocab_size * hidden_dim + v * hidden_dim + d];
    logits[(head * (uint)gid.y + tok) * vocab_size + v] = s;
"""

def medusa_draft_heads(hidden: mx.array, heads_w: mx.array,
                       stream: mx.Stream = mx.cpu) -> mx.array:
    seq_len, hidden_dim = hidden.shape
    num_heads, vocab_size, _ = heads_w.shape
    kernel = mx.fast.metal_kernel(
        name="medusa_heads_kernel",
        input_names=["hidden","heads_w","hidden_dim","vocab_size","num_heads"],
        output_names=["logits"], source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(inputs=[hidden, heads_w,
                         mx.array(hidden_dim,dtype=mx.uint32),
                         mx.array(vocab_size,dtype=mx.uint32),
                         mx.array(num_heads,dtype=mx.uint32)], grid=(vocab_size,seq_len,num_heads),
                 threadgroup=(min(vocab_size,32),1,1),
                 output_shapes=[(num_heads,seq_len,vocab_size)],
                 output_dtypes=[mx.float32], stream=stream)
    return out[0]
