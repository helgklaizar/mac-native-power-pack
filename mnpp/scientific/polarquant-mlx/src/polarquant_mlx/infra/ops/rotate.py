import mlx.core as mx

MSL_SOURCE = """
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    uint in_d  = in_dim_arr[0];
    uint out_d = out_dim_arr[0];
    float s = 0.0f;
    for (uint d = 0; d < in_d; ++d)
        s += w[row * in_d + d] * rot_mat[col * in_d + d];
    out[row * out_d + col] = s;
"""

def polarquant_rotate(w: mx.array, rot_mat: mx.array,
                      stream: mx.Stream = mx.gpu) -> mx.array:
    rows, in_dim = w.shape
    out_dim = rot_mat.shape[0]
    kernel = mx.fast.metal_kernel(
        name="polar_rotate_kernel",
        input_names=["w", "rot_mat", "in_dim_arr", "out_dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[w, rot_mat,
                mx.array([in_dim], dtype=mx.uint32),
                mx.array([out_dim], dtype=mx.uint32)],
        output_shapes=[(rows, out_dim)],
        output_dtypes=[mx.float32],
        grid=(out_dim, rows, 1), threadgroup=(min(out_dim, 32), 1, 1), stream=stream)
    return out[0]
