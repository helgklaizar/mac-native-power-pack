import mlx.core as mx

MSL_SOURCE = """
    uint i = thread_position_in_grid.x;
    uint d = dim_arr[0];
    // SSM step: h = A*h + B*x, y = C*h
    float h_new = A[i] * h_prev[i] + B[i] * x[i % d];
    h_out[i]    = h_new;
    y_out[i]    = C[i] * h_new;
"""

def mamba_ssm_step(x: mx.array, A: mx.array, B: mx.array, C: mx.array,
                   h_prev: mx.array, stream: mx.Stream = mx.gpu) -> tuple:
    d = A.shape[0]
    kernel = mx.fast.metal_kernel(
        name="mamba_ssm_step_kernel",
        input_names=["x", "A", "B", "C", "h_prev", "dim_arr"],
        output_names=["h_out", "y_out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[x, A, B, C, h_prev, mx.array([d], dtype=mx.uint32)],
        output_shapes=[(d,), (d,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(d, 1, 1), threadgroup=(min(d, 256), 1, 1), stream=stream)
    return out[0], out[1]
