# 💎 Best Model & Kernel Builds

This directory contains detailed technical descriptions and performance expectations for the most optimized "builds" in the Mac AI Ecosystem.

## 🚀 The Elite Selection

### Model Architectures
- [Llama-3.2-Metal-Ultra](docs/builds/llama-3-metal.md) — 4-bit optimized instruction following.
*   [DeepSeek-MLA-Native](docs/builds/deepseek-mla.md) — Multi-head Latent Attention with direct Metal kernels.
*   [Qwen-3.5-Turbo-Metal](docs/builds/qwen-turbo.md) — High-throughput coding and reasoning.
*   [Gemma-Vision-Elite](docs/builds/gemma-vision.md) — Native MSL kernels for Vision Transformers.

### Specialized Kernels
*   [Sage-Attention-v2](docs/modules/sage-attention.md) — The fastest attention implementation on Apple Silicon.
*   [Flash-MLA](docs/modules/flash-mla.md) — Native MLA port for DeepSeek V3/R1 logic.
*   [BitNet-1.58b-Native](docs/modules/bitnet-native.md) — Direct ternary weight support on GPU.

## 🛠 Integration Matrix
Each build is validated against the [63-module test suite](README.md#📊-test-results---100-pass-rate).
