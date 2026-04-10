import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.y;
    uint i   = thread_position_in_grid.x;
    uint n   = n_arr[0];
    float s  = 0.0f;
    float inv_sqrt_n = 1.0f / sqrt((float)n);
    for (uint j = 0; j < n; ++j) {
        uint pc = (uint)popcount(i & j);
        float h = (pc % 2 == 0) ? 1.0f : -1.0f;
        s += h * x[tok * n + j];
    }
    out[tok * n + i] = s * inv_sqrt_n;
"""

def quarot_hadamard_transform(x: mx.array, stream: mx.Stream = mx.gpu) -> mx.array:
    if len(x.shape) == 1:
        x = x.reshape([1, -1])
        squeeze = True
    else:
        squeeze = False
    seq_len, n = x.shape
    kernel = mx.fast.metal_kernel(
        name="hadamard_transform_kernel",
        input_names=["x", "n_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[x, mx.array([n], dtype=mx.uint32)],
        output_shapes=[(seq_len, n)],
        output_dtypes=[mx.float32],
        grid=(n, seq_len, 1), threadgroup=(min(n, 32), 1, 1), stream=stream)
    result = out[0]
    return result.reshape([-1]) if squeeze else result
