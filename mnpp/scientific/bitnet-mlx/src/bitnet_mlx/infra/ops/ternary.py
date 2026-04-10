import mlx.core as mx

# Native Metal Shading Language (MSL) implementation of BitNet Linear Layer
# MLX 0.31.1+ API: source = function BODY only, signature is auto-generated.
# Supports ternary quantized weights (+1, 0, -1) stored as int8.

BITNET_SOURCE = """
    uint row = thread_position_in_grid.y;  // batch/seq index
    uint col = thread_position_in_grid.x;  // output feature index

    uint in_f  = in_features[0];
    uint out_f = out_features[0];

    float sum = 0.0f;
    for (uint i = 0; i < in_f; ++i) {
        float x_val = x[row * in_f + i];
        int8_t w_val = w_ternary[col * in_f + i];
        sum += x_val * (float)w_val;
    }

    out[row * out_f + col] = sum * scales[col];
"""

def bitnet_linear(x: mx.array, w_tern: mx.array, scales: mx.array, stream: mx.Stream = mx.gpu) -> mx.array:
    """
    Computes BitNet quantized linear layer using native MSL kernel (MLX 0.31.1+ API).
    w_tern: int8 array of shape [out_features, in_features]
    scales: float32 array of shape [out_features]
    """
    assert w_tern.dtype == mx.int8, "BitNet weights must be int8"

    batch_seq   = x.shape[0] if len(x.shape) > 1 else 1
    in_features  = x.shape[-1]
    out_features = w_tern.shape[0]

    kernel = mx.fast.metal_kernel(
        name="bitnet_linear_kernel",
        input_names=["x", "w_ternary", "scales", "in_features", "out_features"],
        output_names=["out"],
        source=BITNET_SOURCE,
        ensure_row_contiguous=True
    )

    in_arr  = mx.array([in_features],  dtype=mx.uint32)
    out_arr = mx.array([out_features], dtype=mx.uint32)

    out = kernel(
        inputs=[x, w_tern, scales, in_arr, out_arr],
        output_shapes=[x.shape[:-1] + (out_features,)],
        output_dtypes=[mx.float32],
        grid=(out_features, batch_seq, 1),
        threadgroup=(min(out_features, 32), 1, 1),
        stream=stream
    )

    return out[0]
