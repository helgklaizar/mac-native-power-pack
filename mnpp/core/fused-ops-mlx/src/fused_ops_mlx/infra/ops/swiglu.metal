// Fused SwiGLU Kernel
// out = SiLU(x) * y = (x / (1 + exp(-x))) * y
uint i = thread_position_in_grid.x;
float vx = (float)x[i];
float vy = (float)y[i];
float silu = vx / (1.0f + exp(-vx));
out[i] = (T)(silu * vy);
