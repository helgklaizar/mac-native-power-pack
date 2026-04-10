import mlx.core as mx

MSL_SOURCE = """
    uint i   = thread_position_in_grid.x;
    uint dim = dim_arr[0];
    // Titans: neural long-term memory update
    // W_mem <- W_mem - lr * (W_mem * x - x) (simplified gradient step)
    float xi = x[i % dim];
    float lr = lr_arr[0];
    float pred = 0.0f;
    for (uint d = 0; d < dim; ++d)
        pred += W_mem[i / dim * dim + d] * x[d];
    W_out[i] = W_mem[i] - lr * (pred - xi) * x[i % dim];
"""

def titans_neural_memory_step(x: mx.array, W_mem: mx.array,
                               lr: float = 0.01,
                               stream: mx.Stream = mx.gpu) -> mx.array:
    dim = x.shape[0]
    n   = W_mem.shape[0]
    kernel = mx.fast.metal_kernel(
        name="titans_memory_kernel",
        input_names=["x", "W_mem", "lr_arr", "dim_arr"],
        output_names=["W_out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[x, W_mem,
                mx.array([lr], dtype=mx.float32),
                mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1), stream=stream)
    return out[0]
