# 🚀 Mac-Native Power Pack (MNPP)

> **Extreme-performance Metal kernels for MLX. Optimized for Apple Silicon.**

MNPP is the quintessence of optimizations for local inference. We have stripped away everything non-essential, leaving 7 elite kernels that provide maximum speed and efficiency gains on Mac unified memory.

## 💎 The Elite Seven
1. **Sage-Attention** — The fastest attention kernel on Metal.
2. **Turbo-Quant** — Leader in weight dequantization throughput.
3. **Flash-Attention** — Uncompromising performance with long contexts.
4. **Speculative-Decode** — Up to 2-3x faster generation via predictive algorithms.
5. **Paged-Attention** — KV-cache optimization to maximize RAM efficiency.
6. **FlashMLA** — Native port of Multi-head Latent Attention (DeepSeek V3/R1).
7. **BitNet-Native** — Direct support for ternary (1.58-bit) weights on GPU.

## 📦 Project Structure
- `core/` — Top 5 modules for daily inference acceleration.
- `scientific/` — Foundation for 1-bit and experimental models.
- `nouveau_2026/` — The cutting edge: DeepSeek MLA architecture support.

## 🛠 Usage
Each module is a standalone logic block in `/src/infra/ops`. 
Integrate into your project via `mlx.fast.metal_kernel` for native execution without Python overhead.

---
*Pure Speed. No Bloat. Apple Silicon Native. 2026.*

