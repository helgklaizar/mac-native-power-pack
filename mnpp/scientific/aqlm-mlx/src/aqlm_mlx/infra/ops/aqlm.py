import mlx.core as mx

# Native Metal Shading Language (MSL) implementation of AQLM Dequantization
# MLX 0.31.1+ API: source = function BODY only, signature is auto-generated.

AQLM_DEQUANT_SOURCE = """
    uint group_idx = thread_position_in_grid.x;

    uint num_books  = config[0];
    uint book_size  = config[1];
    uint group_size = config[2];

    for (uint g = 0; g < group_size; ++g) {
        float sum = 0.0f;
        for (uint b = 0; b < num_books; ++b) {
            uint code_val = codes[group_idx * num_books + b];
            uint cb_idx   = (b * book_size * group_size) + (code_val * group_size) + g;
            sum += codebooks[cb_idx];
        }
        out[group_idx * group_size + g] = sum;
    }
"""

def dequantize_aqlm(codes: mx.array, codebooks: mx.array,
                    stream: mx.Stream = mx.gpu) -> mx.array:
    """
    Dequantizes AQLM compressed weight matrices directly on Apple GPU (MLX 0.31.1+ API).
    codes:     uint8 [M, num_books]
    codebooks: float32 [num_books, book_size, group_size]
    Returns:   float32 [M, group_size]
    """
    assert codes.dtype == mx.uint8, "codes must be uint8"
    assert len(codebooks.shape) == 3, "codebooks must be 3D"

    M = codes.shape[0]
    num_books, book_size, group_size = codebooks.shape

    kernel = mx.fast.metal_kernel(
        name="aqlm_dequantize",
        input_names=["codes", "codebooks", "config"],
        output_names=["out"],
        source=AQLM_DEQUANT_SOURCE,
        ensure_row_contiguous=True
    )

    config = mx.array([num_books, book_size, group_size], dtype=mx.uint32)

    out = kernel(
        inputs=[codes, codebooks, config],
        output_shapes=[(M, group_size)],
        output_dtypes=[mx.float32],
        grid=(M, 1, 1),
        threadgroup=(min(M, 256), 1, 1),
        stream=stream
    )

    return out[0]
