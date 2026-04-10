import mlx.core as mx

MSL_SOURCE = """
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    uint dim = dim_arr[0];
    float val = 0.0f;
    for (uint d = 0; d < dim; ++d)
        val += w[row * dim + d] * R[col * dim + d];
    val *= scale[col];
    out[row * dim + col] = val;
"""

def spinquant_rotate_quantize(w: mx.array, R: mx.array, scale: mx.array,
                               stream: mx.Stream = mx.gpu) -> mx.array:
    rows, dim = w.shape
    kernel = mx.fast.metal_kernel(
        name="spinquant_rotate_kernel",
        input_names=["w", "R", "scale", "dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[w, R, scale, mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(rows, dim)],
        output_dtypes=[mx.float32],
        grid=(dim, rows, 1), threadgroup=(min(dim, 32), 1, 1), stream=stream)
    return out[0]
