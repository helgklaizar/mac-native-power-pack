This document tracks identified native Metal/MLX modules for potential integration into the MNPP (Mac-Native Power Pack).

*   **[Detailed Build Specs](docs/builds.md)** — In-depth analysis of the Elite Seven.
*   **[Experimental Logs](experiments/LOGS.md)** — Real-world benchmarks and test reports.

## 🚀 Priority Targets for Integration

| Module Name | Source / Repo Link | Focus / Feature |
| :--- | :--- | :--- |
| **BitNet-MLX** | [exo-explore/mlx-bitnet](https://github.com/exo-explore/mlx-bitnet) | 1.58-bit ternary quantization kernels |
| **Mamba-MLX** | [alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py) | SSM (State Space Model) Metal implementation |
| **Metal-Flash-Attention** | [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) | SOTA performance optimizations for Apple Silicon |
| **DeepSeek-V3-MLA** | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | Reference for Multi-head Latent Attention (MLA) |
| **MLX-Examples** | [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) | Official reference kernels for Llama, Whisper, etc. |
| **MLX-VLM** | [ml-explore/mlx-vlm](https://github.com/ml-explore/mlx-vlm) | Vision-Language Models for Apple Silicon |
| **Awesome-MLX-Hub** | [ml-explore/mlx-examples/discussions/1411](https://github.com/ml-explore/mlx-examples/discussions/1411) | Official community directory of 80+ projects |
| **MLX-Benchmark** | [TristanBilot/mlx-benchmark](https://github.com/TristanBilot/mlx-benchmark) | Benchmarking operations across Apple M-series chips |
| **MLX-LM-Bench** | [gauri-nagavkar/mlxlm_bench](https://github.com/gauri-nagavkar/mlxlm_bench) | Inference performance metrics for MLX models |
| **MLX-BitNet-minGPT** | [adi-dhulipala/mlx-bitnet-mingpt](https://github.com/adi-dhulipala/mlx-bitnet-mingpt) | Educational ternary linear layer implementation |


## 🏺 Reference Model Builds (MLX-Community)

| Model Name | Hugging Face Link | Specs |
| :--- | :--- | :--- |
| **Qwen-3.5-9B-4bit** | [mlx-community/Qwen3.5-9B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit) | Balanced coding/reasoning |
| **Llama-3.2-1B-4bit** | [mlx-community/Llama-3.2-1B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit) | Lightweight instruction follow |
| **Gemma-4-31b-4bit** | [mlx-community/gemma-4-31b](https://huggingface.co/mlx-community/gemma-4-31b) | SOTA Multimodal (Vision/Text) |
| **GLM-5.1-4bit** | [mlx-community/GLM-5.1-4bit](https://huggingface.co/mlx-community/GLM-5.1-4bit) | Large-scale parameter test |

## 🛠 Integration Plan

1.  **Extract Kernels**: Isolate the pure MSL (`.metal`) and the `mlx.fast.metal_kernel` wrappers.
2.  **MNPP Standardization**: Move to `mnpp/<category>/<module-name>/src/...`.
3.  **Benchmarking**: Run throughput tests on Apple Silicon (M-series).
4.  **Registration**: Add to `eco.yaml` for CLI management.
