import mlx.core as mx

MSL_SOURCE = """
    uint i   = thread_position_in_grid.x;
    uint dim = dim_arr[0];
    // Update compressive memory: M = M + outer(v, sigma(k))
    // Here simplified: M[i] += v[i] * mean(k)
    float k_mean = 0.0f;
    for (uint d = 0; d < dim; ++d) k_mean += new_k[d];
    k_mean /= (float)dim;
    out_M[i] = mem_M[i] + new_v[i % dim] * k_mean;
    out_z[i] = mem_z[i] + k_mean;
"""

def infini_attention_update(mem_M: mx.array, mem_z: mx.array,
                             new_k: mx.array, new_v: mx.array,
                             stream: mx.Stream = mx.gpu) -> tuple[mx.array, mx.array]:
    """Infini-Attention: update compressive memory with new KV."""
    dim = new_k.shape[0]
    n   = mem_M.shape[0]
    kernel = mx.fast.metal_kernel(
        name="infini_attn_update_kernel",
        input_names=["mem_M", "mem_z", "new_k", "new_v", "dim_arr"],
        output_names=["out_M", "out_z"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[mem_M, mem_z, new_k, new_v, mx.array([dim], dtype=mx.uint32)],
        output_shapes=[(n,), (n,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        stream=stream
    )
    return out[0], out[1]
