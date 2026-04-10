import mlx.core as mx

MSL_SOURCE = """
// SpinQuant: random rotation + quantization for LLM weight reduction
    uint row = gid.y; uint col = gid.x;
    float s = 0.0f;
    for (uint d = 0; d < in_dim; ++d)
        s += w[row * in_dim + d] * R[col * in_dim + d];
    // Quantize rotated weight
    float sc = scale[row];
    qout[row * out_dim + col] = (int8_t)clamp(round(s / sc), -127.0f, 127.0f);
"""

def spinquant_rotate_quantize(w: mx.array, R: mx.array, scale: mx.array,
                               stream: mx.Stream = mx.cpu) -> mx.array:
    rows, in_dim = w.shape
    out_dim = R.shape[0]
    kernel = mx.fast.metal_kernel(
        name="spin_rotate_quant",
        input_names=["w","R","scale","in_dim","out_dim"], output_names=["qout"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(inputs=[w, R, scale,
                          mx.array(in_dim,dtype=mx.uint32),
                          mx.array(out_dim,dtype=mx.uint32)], grid=(out_dim,rows,1),
                 threadgroup=(min(out_dim,32),1,1),
                 output_shapes=[(rows,out_dim)], output_dtypes=[mx.int8], stream=stream)
    return out[0]
