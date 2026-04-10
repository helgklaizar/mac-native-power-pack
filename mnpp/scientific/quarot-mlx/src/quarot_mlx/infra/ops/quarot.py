import mlx.core as mx

MSL_SOURCE = """
// QuaRot: Hadamard rotation for quantization-friendly activation transformation
    uint tok = gid.y; uint i = gid.x;
    float s = 0.0f;
    float sign = 1.0f;
    // Walsh-Hadamard: H[i][j] = (-1)^popcount(i & j) / sqrt(n)
    for (uint j = 0; j < n; ++j) {
        uint pc = (uint)popcount(i & j);
        float h = (pc % 2 == 0) ? 1.0f : -1.0f;
        s += h * x[tok * n + j];
    }
    out[tok * n + i] = s / sqrt((float)n);
"""

def quarot_hadamard_transform(x: mx.array, stream: mx.Stream = mx.cpu) -> mx.array:
    seq_len, n = x.shape
    kernel = mx.fast.metal_kernel(
        name="hadamard_transform_kernel",
        input_names=["x","n"], output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(inputs=[x, mx.array(n,dtype=mx.uint32)], grid=(n,seq_len,1),
                 threadgroup=(min(n,256),1,1),
                 output_shapes=[x.shape], output_dtypes=[mx.float32], stream=stream)
    return out[0]
