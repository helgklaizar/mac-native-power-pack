import mlx.core as mx

MSL_SOURCE = """
// Speculative Decode: verify draft tokens against target logits
    uint t = gid.x;
    uint tok = draft_tokens[t];
    float p_draft  = draft_logits[t * vocab_size + tok];
    float p_target = target_logits[t * vocab_size + tok];
    // Acceptance criterion: target_prob >= draft_prob
    float ratio = exp(p_target - p_draft);
    float threshold = min(1.0f, ratio);
    accepted[t] = (threshold >= 0.5f) ? 1 : 0;
"""

def speculative_verify(draft_logits: mx.array, target_logits: mx.array,
                        draft_tokens: mx.array, stream: mx.Stream = mx.cpu) -> mx.array:
    n, vocab_size = draft_logits.shape
    kernel = mx.fast.metal_kernel(
        name="spec_verify_kernel",
        input_names=["draft_logits","target_logits","draft_tokens","vocab_size"],
        output_names=["accepted"], source=MSL_SOURCE, ensure_row_contiguous=True)
    out = kernel(inputs=[draft_logits, target_logits, draft_tokens,
                          mx.array(vocab_size,dtype=mx.uint32)], grid=(n,1,1), threadgroup=(256,1,1),
                 output_shapes=[(n,)], output_dtypes=[mx.uint8], stream=stream)
    return out[0]
