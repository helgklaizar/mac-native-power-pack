import mlx.core as mx

MSL_SOURCE = """
// TITANS: neural long-term memory with surprise-based update
    uint tok = gid.y; uint m = gid.x;
    float pred = 0.0f;
    for (uint d = 0; d < in_dim; ++d)
        pred += x[tok * in_dim + d] * W_mem[m * in_dim + d];
    float target = x[tok * in_dim + (m % in_dim)];
    float err = target - pred;
    surprise[tok * mem_dim + m] = err * err;
    mem_out[tok * mem_dim + m] = pred;
"""

def titans_neural_memory_step(x: mx.array, W_mem: mx.array,
                               stream: mx.Stream = mx.cpu) -> tuple:
    seq_len, in_dim = x.shape
    mem_dim = W_mem.shape[0]
    kernel = mx.fast.metal_kernel(
        name="titans_mem_kernel",
        input_names=["x","W_mem","mem_dim","in_dim"],
        output_names=["mem_out","surprise"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(inputs=[x, W_mem,
                          mx.array(mem_dim,dtype=mx.uint32),
                          mx.array(in_dim,dtype=mx.uint32)], grid=(mem_dim,seq_len,1),
                 threadgroup=(min(mem_dim,32),1,1),
                 output_shapes=[(seq_len,mem_dim),(seq_len,mem_dim)],
                 output_dtypes=[mx.float32,mx.float32], stream=stream)
    return out[0], out[1]
