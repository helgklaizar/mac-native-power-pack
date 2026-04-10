# 🔭 MNPP Vision: The Fused Infinity Core

## 🌍 The Problem
Current MLX projects are fragmented. One repo has a fast Attention kernel, another has a Paged Attention implementation, and a third has BitNet support. Developers have to manually stitch these together, and often there is no **Auto-Tuning** for specific M-series chips (M1 vs. M4).

## 🚀 Our "Better" Solution: MNPP Unified Framework
We don't just collect kernels; we **fuse** and **automate** them.

### 1. Unified Fused Blocks (UFB)
Instead of calling `RMSNorm -> QKV -> Rotary -> Attention` as separate operations, we develop **Fused Metal Kernels** that execute the entire block in minimal GPU passes. This reduces memory bandwidth pressure—the primary bottleneck on Apple Silicon.

### 2. Micro-Kernel Auto-Tuning
A runtime profiler that detects the hardware (e.g., M2 Max vs M1 Pro) and dynamically selects the optimal `threadgroup` sizes and `simd_group` counts for each kernel.

### 3. The "Drop-In" Replacement
MNPP will provide a wrapper that can replace standard `mlx.nn` layers in existing libraries (like `mlx-lm`) with our optimized Metal versions without changing the user's model logic.

## 🏗 Roadmap to Superiority
- [ ] **Phase 1: The Fused Norm-Gate**: Fusing RMSNorm and Silu-based Gating (LLama/Qwen style) into one kernel.
- [ ] **Phase 2: MLA Native Core**: Implementing the world's most optimized Flash-MLA for DeepSeek.
- [ ] **Phase 3: 1.58-bit Ternary Engine**: A production-ready inference engine for BitNet kernels.

## 🏆 Competitive Advantage
| Feature | Existing Repos | **MNPP (Our Build)** |
| :--- | :--- | :--- |
| **Fusion** | Atomic / Piecewise | Block-Level Fused |
| **Optimization** | Static / General | Hardware-Aware Auto-Tuning |
| **Integration** | Manual / Hard | CLI-based Hot-Swap |
| **Support** | Older Architectures | Nouveau 2026 (MLA, BitNet) |
