# 🧠 Memory Whisperer V1 (RAM Compression Build)

This assembly is dedicated to running large-scale LLMs (30B - 70B+) on **entry-level Apple Silicon hardware** with limited physical RAM (8GB - 16GB).

## 🧩 Components
- **Kernel Strategy**: [BitNet 1.58b](../../mnpp/scientific/bitnet-mlx/) — Native ternary weight logic reducing memory footprint by up to 3-4x vs. standard 4-bit.
- **Caching Engine**: [omlx](../../REGISTRY.md) — SSD Swap optimization for continuous batching.
- **Quantization**: 1.58-bit (Ternary) for weights / 8-bit for activations.

## 📊 Expected Results (M1/M2/M3 with 8GB RAM)

| Model Size | 4-bit Logic | **MNPP Memory Whisperer** | Status |
| :--- | :--- | :--- | :--- |
| **8B (Llama 3)** | Runs (Tight) | **Runs (Smooth)** | ✅ Validated |
| **30B (Command R)** | Fails (OOM) | **Runs (Swap-active)** | ⚠️ Experimental |
| **70B (Llama 3)** | Fails | **Potential (Swap-heavy)** | 🔭 Frontier |

## 🏆 Best Applied In:
1.  **MacBook Air / Mini Base Models**: Perfect for users who bought the 8GB RAM version but want to run modern SOTA models locally.
2.  **Resource-Constrained Environments**: Edge devices or shared CI runners.
3.  **Experimental 1-bit ML**: Development of native ternary architectures.

---
*Created by MNPP "Beyond Limits" Initiative.*
