import mlx.core as mx

MSL_SOURCE = """
    uint idx       = thread_position_in_grid.x;
    uint inner_dim = inner_dim_arr[0];
    uint ch        = idx / inner_dim;
    float s = scale[ch];
    float z = zero[ch];
    out[idx] = ((float)((int)qw[idx]) - z) * s;
"""

def hqq_dequantize(qw: mx.array, scale: mx.array, zero: mx.array,
                   stream: mx.Stream = mx.gpu) -> mx.array:
    """HQQ half-quadratic quantization dequantize: float = (q - zero) * scale."""
    assert len(qw.shape) == 2
    num_ch, inner = qw.shape
    kernel = mx.fast.metal_kernel(
        name="hqq_dequant_kernel",
        input_names=["qw", "scale", "zero", "inner_dim_arr"],
        output_names=["out"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[qw, scale, zero, mx.array([inner], dtype=mx.uint32)],
        output_shapes=[(num_ch, inner)],
        output_dtypes=[mx.float32],
        grid=(num_ch * inner, 1, 1),
        threadgroup=(min(num_ch * inner, 256), 1, 1),
        stream=stream
    )
    return out[0]
