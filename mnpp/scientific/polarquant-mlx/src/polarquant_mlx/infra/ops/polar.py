import mlx.core as mx

MSL_SOURCE = """
// PolarQuant: rotates weights into a polar-friendly basis before quantization
    uint row = gid.y; uint col = gid.x;
    float s = 0.0f;
    for (uint d = 0; d < in_dim; ++d)
        s += w[row * in_dim + d] * rot_mat[col * in_dim + d];
    out[row * out_dim + col] = s;
"""

def polarquant_rotate(w: mx.array, rot_mat: mx.array,
                      stream: mx.Stream = mx.cpu) -> mx.array:
    rows, in_dim = w.shape
    out_dim = rot_mat.shape[0]
    kernel = mx.fast.metal_kernel(
        name="polar_rotate_kernel",
        input_names=["w","rot_mat","in_dim","out_dim"], output_names=["out"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(inputs=[w, rot_mat,
                         mx.array(in_dim,dtype=mx.uint32),
                         mx.array(out_dim,dtype=mx.uint32)], grid=(out_dim,rows,1),
                 threadgroup=(min(out_dim,32),1,1),
                 output_shapes=[(rows,out_dim)], output_dtypes=[mx.float32], stream=stream)
    return out[0]
