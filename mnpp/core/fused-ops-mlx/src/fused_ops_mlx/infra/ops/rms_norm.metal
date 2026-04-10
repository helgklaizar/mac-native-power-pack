// Fused RMSNorm Body (JIT-injected into MLX custom kernel)
uint row = thread_position_in_grid.x;
const device T* x = input + row * D;
device T* y = output + row * D;

// Phase 1: Sum squares
float sum_sq = 0.0f;
for (uint i = 0; i < (uint)D; ++i) {
    float val = (float)x[i];
    sum_sq += val * val;
}

// Phase 2: Compute RMS inverse
float rms_inv = rsqrt(sum_sq / (float)D + epsilon);

// Phase 3: Normalize and scale
for (uint i = 0; i < (uint)D; ++i) {
    y[i] = (T)((float)x[i] * rms_inv * (float)weight[i]);
}
