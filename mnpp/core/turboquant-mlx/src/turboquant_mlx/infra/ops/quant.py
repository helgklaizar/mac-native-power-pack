import mlx.core as mx

QUANT_SOURCE = """
    uint ch   = thread_position_in_grid.y;
    uint idx  = thread_position_in_grid.x;
    uint inner = inner_dim_arr[0];
    uint flat  = ch * inner + idx;
    float q    = (input[flat] / scale[ch]) + zero_p[ch];
    if (q < -128.0f) q = -128.0f;
    if (q >  127.0f) q =  127.0f;
    output[flat] = (int8_t)round(q);
"""

DEQUANT_SOURCE = """
    uint ch    = thread_position_in_grid.y;
    uint idx   = thread_position_in_grid.x;
    uint inner = inner_dim_arr[0];
    uint flat  = ch * inner + idx;
    output[flat] = ((float)input[flat] - zero_p[ch]) * scale[ch];
"""

def quantize(x: mx.array, scale: mx.array, zero_point: mx.array,
             stream: mx.Stream = mx.gpu) -> mx.array:
    """TurboQuant: float32 → int8 per-channel quantization."""
    assert len(x.shape) == 2
    num_ch, inner = x.shape
    kernel = mx.fast.metal_kernel(
        name="turboquant_quant_int8",
        input_names=["input", "scale", "zero_p", "inner_dim_arr"],
        output_names=["output"],
        source=QUANT_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[x, scale, zero_point, mx.array([inner], dtype=mx.uint32)],
        output_shapes=[(num_ch, inner)],
        output_dtypes=[mx.int8],
        grid=(inner, num_ch, 1),
        threadgroup=(min(inner, 256), 1, 1),
        stream=stream
    )
    return out[0]

def dequantize(x: mx.array, scale: mx.array, zero_point: mx.array,
               stream: mx.Stream = mx.gpu) -> mx.array:
    """TurboQuant: int8 → float32 per-channel dequantization."""
    assert len(x.shape) == 2
    num_ch, inner = x.shape
    kernel = mx.fast.metal_kernel(
        name="turboquant_dequant_int8",
        input_names=["input", "scale", "zero_p", "inner_dim_arr"],
        output_names=["output"],
        source=DEQUANT_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[x, scale, zero_point, mx.array([inner], dtype=mx.uint32)],
        output_shapes=[(num_ch, inner)],
        output_dtypes=[mx.float32],
        grid=(inner, num_ch, 1),
        threadgroup=(min(inner, 256), 1, 1),
        stream=stream
    )
    return out[0]
