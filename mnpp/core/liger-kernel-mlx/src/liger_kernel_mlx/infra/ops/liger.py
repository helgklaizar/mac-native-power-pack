import mlx.core as mx

MSL_SOURCE = """
    uint tok   = thread_position_in_grid.x;
    uint vsize = vocab_size_arr[0];
    uint base  = tok * vsize;
    uint label = labels[tok];

    float max_v = logits[base];
    for (uint v = 1; v < vsize; ++v)
        if (logits[base + v] > max_v) max_v = logits[base + v];

    float sum_exp = 0.0f;
    for (uint v = 0; v < vsize; ++v)
        sum_exp += exp(logits[base + v] - max_v);

    losses[tok] = log(sum_exp) + max_v - logits[base + label];
"""

def liger_cross_entropy(logits: mx.array, labels: mx.array,
                        stream: mx.Stream = mx.gpu) -> mx.array:
    """Liger fused cross-entropy (stable log-softmax + NLL)."""
    assert len(logits.shape) == 2
    seq_len, vocab_size = logits.shape
    assert labels.dtype == mx.uint32
    kernel = mx.fast.metal_kernel(
        name="liger_ce_kernel",
        input_names=["logits", "labels", "vocab_size_arr"],
        output_names=["losses"],
        source=MSL_SOURCE,
        ensure_row_contiguous=True
    )
    out = kernel(
        inputs=[logits, labels, mx.array([vocab_size], dtype=mx.uint32)],
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.float32],
        grid=(seq_len, 1, 1),
        threadgroup=(1, 1, 1),
        stream=stream
    )
    return out[0]
