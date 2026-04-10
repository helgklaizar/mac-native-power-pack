import mlx.core as mx

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
