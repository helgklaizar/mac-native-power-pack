import mlx.core as mx

MSL_SOURCE = """
    uint tok = thread_position_in_grid.x;
    uint vs  = vocab_size_arr[0];
    uint base = tok * vs;
    // Accept if argmax(draft) == argmax(target)
    uint draft_top = 0; float draft_max = draft_logits[base];
    uint target_top = 0; float target_max = target_logits[base];
    for (uint v = 1; v < vs; ++v) {
        if (draft_logits[base+v]  > draft_max)  { draft_max  = draft_logits[base+v];  draft_top  = v; }
        if (target_logits[base+v] > target_max) { target_max = target_logits[base+v]; target_top = v; }
    }
    accept[tok] = (draft_top == target_top) ? (uint8_t)1 : (uint8_t)0;
"""

def speculative_verify(draft_logits: mx.array, target_logits: mx.array,
                       stream: mx.Stream = mx.gpu) -> mx.array:
    seq_len, vocab_size = draft_logits.shape
    kernel = mx.fast.metal_kernel(
        name="speculative_verify_kernel",
        input_names=["draft_logits", "target_logits", "vocab_size_arr"],
        output_names=["accept"],
        source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(
        inputs=[draft_logits, target_logits,
                mx.array([vocab_size], dtype=mx.uint32)],
        output_shapes=[(seq_len,)],
        output_dtypes=[mx.uint8],
        grid=(seq_len, 1, 1), threadgroup=(1, 1, 1), stream=stream)
    return out[0]
