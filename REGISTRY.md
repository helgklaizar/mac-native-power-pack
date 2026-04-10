# 🔍 Eco-Metal Discovery & Registry Index

This document tracks identified native Metal/MLX modules for potential integration into the MNPP (Mac-Native Power Pack).

## 🚀 Priority Targets for Integration

| Module Name | Source / Repo Link | Focus / Feature |
| :--- | :--- | :--- |
| **BitNet-MLX** | [exo-explore/mlx-bitnet](https://github.com/exo-explore/mlx-bitnet) | 1.58-bit ternary quantization kernels |
| **Mamba-MLX** | [alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py) | SSM (State Space Model) Metal implementation |
| **Metal-Flash-Attention** | [philip-turner/metal-flash-attention](https://github.com/philip-turner/metal-flash-attention) | SOTA performance optimizations for Apple Silicon |
| **DeepSeek-V3-MLA** | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | Reference for Multi-head Latent Attention (MLA) |
| **MLX-Examples** | [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) | Official reference kernels for Llama, Whisper, etc. |
| **Awesome-MLX** | [replicate/awesome-mlx](https://github.com/replicate/awesome-mlx) | Curated collection of MLX-native models and tools |
| **Liger-Kernel-Ref** | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Fused kernels (RMSNorm, Softmax) for porting to Metal |
| **MLX-BitNet-minGPT** | [adi-dhulipala/mlx-bitnet-mingpt](https://github.com/adi-dhulipala/mlx-bitnet-mingpt) | Educational ternary linear layer implementation |


## 🛠 Integration Plan

1.  **Extract Kernels**: Isolate the pure MSL (`.metal`) and the `mlx.fast.metal_kernel` wrappers.
2.  **MNPP Standardization**: Move to `mnpp/<category>/<module-name>/src/...`.
3.  **Benchmarking**: Run throughput tests on Apple Silicon (M-series).
4.  **Registration**: Add to `eco.yaml` for CLI management.
