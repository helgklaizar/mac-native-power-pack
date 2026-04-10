# 🧪 MNPP Experimental Logs & Benchmarks

This document tracks local and reference benchmark data for MLX operations across various Apple Silicon chips.

## 🚀 Baseline: MLX Operations (Raw Metal)
*Source: [TristanBilot/mlx-benchmark](https://github.com/TristanBilot/mlx-benchmark)*

### Summary Matrix (Latency in ms)

| Operation (dim=1024x1024) | M1 Ultra | M2 Max | M3 Max | Speedup (M3 vs M1) |
| :--- | :--- | :--- | :--- | :--- |
| **MatMul** | 1.25 | 1.36 | 0.86 | +45% |
| **Softmax** | 3.41 | 5.61 | 2.98 | +14% |
| **Linear** | 1.67 | 3.80 | 0.78 | +114% |
| **BCE Loss** | 0.34 | 0.24 | 0.27 | +25% |

## 🧪 Active Experiments

### [EXP-001] FlashMLA vs. Sparse Attention
- **Status**: In-Progress
- **Focus**: Comparing throughput with Million-token contexts.
- **Reference**: [DeepSeek V3 MLA Reference](https://github.com/deepseek-ai/DeepSeek-V3)

### [EXP-002] BitNet Ternary Kernel Throughput
- **Status**: Validated
- **Focus**: Impact of 1.58-bit quantization on inference latency vs accuracy.
- **Log**: [experiments/exp-002.log](experiments/exp-002.log)
