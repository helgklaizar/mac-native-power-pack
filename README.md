<div align="center">

# 🌿 Eco-Metal
### The Apple Silicon AI Kernel Ecosystem

[![MLX](https://img.shields.io/badge/MLX-0.31.1+-orange?style=for-the-badge&logo=apple)](https://github.com/ml-explore/mlx)
[![Tests](https://img.shields.io/badge/Tests-63%2F63_PASS-brightgreen?style=for-the-badge)](tests/)
[![Hardware](https://img.shields.io/badge/Apple_Silicon-M_Series-black?style=for-the-badge&logo=apple)](https://apple.com/silicon)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)

**63 production-validated Metal/MLX kernel modules for extreme AI performance on Apple Silicon.**  
No CUDA. No compromises. Native Metal Shading Language from the ground up.

</div>

---

## 🔥 What is Eco-Metal?

Eco-Metal is a **modular, registry-based collection of 63 high-performance AI kernel implementations** — all written natively for Apple Silicon using **MLX + Metal Shading Language (MSL)**. Each module is an independently loadable, JIT-compiled GPU kernel covering the full spectrum of modern LLM inference optimization:

- ⚡ **Extreme quantization** — BitNet, AQLM, HQQ, TurboQuant, QuaRot, SpinQuant, OmniQuant, MXFP4, PolarQuant
- 🧠 **Attention variants** — Flash Attention 3, Sage Attention, MLA, Ring Attention, Infini-Attention, NSA, Spotlight, Tri-Attention, MIMO-Flash
- 🗜️ **KV-Cache compression** — Rocket-KV, H2O, KVTC, Evol-KV, Pyramid-KV, KVComp, Paged Attention, Radix Cache, Tiered Cache
- 🚀 **Speculative decoding** — Medusa, EAGLE, Lookahead, Self-Spec, Speculative Decode, Search Decode, Plan-and-Explore
- 🔧 **System-level** — Metal Scheduler, LM Cache, Streaming, Context Bridge, Forge (Graph Dispatcher)

---

## 📊 Test Results — 100% Pass Rate

All 63 modules validated on Apple Silicon (MLX 0.31.1, Metal backend). Tests run in 3 batches covering MSL kernel correctness, GPU execution, and live Llama inference.

| Batch | Modules Tested | Passed | Failed | Avg Latency |
|:------|:--------------:|:------:|:------:|:-----------:|
| **Batch 1** — Core MSL Kernels | 6 + 1 inference | **7 / 7** | 0 | ~25–55 ms |
| **Batch 2** — Extended Modules | 19 | **19 / 19** | 0 | ~1–52 ms |
| **Batch 3** — Full Sweep | 37 | **37 / 37** | 0 | ~1–8 ms |
| **Total** | **63** | ✅ **63 / 63** | **0** | — |

### 🧪 Inference Benchmark (Llama-3.2-1B-4bit on Apple Silicon)
```
Model    : mlx-community/Llama-3.2-1B-Instruct-4bit
Latency  : 408 ms  
Throughput: 85.7 tok/s  
Status   : ✅ PASS
```

---

## 📦 Module Catalog (63 Modules)

<details>
<summary><strong>🔬 Quantization (11 modules)</strong></summary>

| Module | Description |
|:-------|:------------|
| `turboquant-mlx` | Extreme multi-bit weight quantization |
| `bitnet-mlx` | Ternary 1.58-bit linear layers via Metal MSL |
| `aqlm-mlx` | Additive Quantization of Language Models (dequantize kernel) |
| `hqq-mlx` | Half-Quadratic Quantization with Metal backend |
| `omniquant-mlx` | Omnidirectional calibration-based quantization |
| `quarot-mlx` | Rotation-based weight quantization |
| `spinquant-mlx` | Learned rotation quantization |
| `polarquant-mlx` | Polar decomposition quantization |
| `qjl-quant-mlx` | QJL random projection quantization |
| `qaq-mlx` | Quality-Adaptive Quantization |
| `mxfp4-mlx` | MX FP4 microscaling format |
| `super-weight-mlx` | Super-weight outlier protection |

</details>

<details>
<summary><strong>⚡ Attention Variants (14 modules)</strong></summary>

| Module | Description |
|:-------|:------------|
| `flash-attention-mlx` | Flash Attention 3 — fused QKV Metal kernel |
| `flash-infer-mlx` | FlashInfer-style decode kernel |
| `sage-attention-mlx` | SAGE sparse + approximate attention |
| `mla-mlx` | Multi-head Latent Attention (DeepSeek style) |
| `ring-attention-mlx` | Ring Attention for distributed long-context |
| `infini-attention-mlx` | Compressive memory attention |
| `nsa-attention-mlx` | Native Sparse Attention |
| `spotlight-attention-mlx` | Spotlight local + global window attention |
| `tri-attention-mlx` | Tri-attention decomposition |
| `mimo-flash-mlx` | Multi-In Multi-Out Flash variant |
| `star-attention-mlx` | Star-topology context sharing |
| `attention-fuse-mlx` | Fused QKV + RoPE Metal kernel |
| `attention-matching-mlx` | Score-threshold attention masking |
| `minference-mlx` | Million-token sparse inference |

</details>

<details>
<summary><strong>🗜️ KV-Cache & Memory (10 modules)</strong></summary>

| Module | Description |
|:-------|:------------|
| `rocket-kv-mlx` | Rocket-KV high-speed cache eviction |
| `h2o-mlx` | Heavy Hitter Oracle cache compression |
| `kvtc-mlx` | KV token compression |
| `evol-kv-mlx` | Evolutionary KV pruning |
| `pyramid-kv-mlx` | Pyramid-shaped hierarchical cache |
| `kvcomp-mlx` | KV delta compression |
| `paged-attention-mlx` | vLLM-style paged KV memory |
| `radix-cache-mlx` | Radix-tree prefix KV cache |
| `tiered-cache-mlx` | Tiered VRAM / system memory cache |
| `lm-cache-mlx` | LM Cache persistent disk KV store |

</details>

<details>
<summary><strong>🚀 Speculative Decoding (7 modules)</strong></summary>

| Module | Description |
|:-------|:------------|
| `eagle-mlx` | EAGLE draft-model speculative decoding |
| `medusa-mlx` | Medusa multi-head parallel decoding |
| `lookahead-mlx` | Lookahead decoding n-grams |
| `self-spec-mlx` | Self-speculative early-exit draft |
| `speculative-decode-mlx` | Classic draft/verify speculative decode |
| `search-decode-mlx` | Tree-search guided decode |
| `plan-and-explore-mlx` | Planning with exploration rollout |

</details>

<details>
<summary><strong>🔧 System & Infra (9 modules)</strong></summary>

| Module | Description |
|:-------|:------------|
| `forge-mlx` | Compute graph dispatcher |
| `metal-scheduler-mlx` | Metal command queue scheduler |
| `streaming-mlx` | Token streaming buffer |
| `context-bridge-mlx` | Cross-model context bridge |
| `context-engineering-mlx` | Context window engineering |
| `content-prefix-mlx` | Prefix caching logic |
| `multi-agent-orchestrator-mlx` | Multi-agent coordinator |
| `cuda-bridge-mlx` | CUDA→Metal translation layer |
| `triton-bridge-mlx` | Triton kernel→Metal bridge |

</details>

<details>
<summary><strong>🧬 Advanced / MoE (6 modules + others)</strong></summary>

| Module | Description |
|:-------|:------------|
| `fused-moe-mlx` | Fused Mixture-of-Experts routing + compute |
| `expert-selective-mlx` | Expert selection gating |
| `mamba-mlx` | Mamba SSM state-space kernel |
| `titans-mlx` | TITANS memory architecture |
| `tome-mlx` | Token Merging for ViT/LLM |
| `layerskip-mlx` | Early-exit layer skipping |
| `inference-time-compute-mlx` | Test-time compute scaling |
| `block-sparse-mlx` | Block-sparse Matrix × Matrix Metal kernel |
| `liger-kernel-mlx` | Fused cross-entropy, RMS norm, SwiGLU |
| `apple-sharing-mlx` | Metal buffer sharing across processes |
| `verifiable-rewards-mlx` | RLHF verifiable reward shaping |

</details>

---

## 🏗 Architecture

```
eco-metal/
├── modules/
│   └── Metal/               # 63 independent MLX kernel modules
│       ├── flash-attention-mlx/
│       │   └── src/         # MSL .metal + Python MLX wrapper
│       ├── bitnet-mlx/
│       ├── turboquant-mlx/
│       └── ...              # (60 more modules)
├── tests/
│   ├── test_metal_modules.py          # Batch 1 — Core MSL kernels
│   ├── test_metal_modules_extended.py # Batch 2 — Extended coverage
│   ├── test_metal_batch3.py           # Batch 3 — Full sweep
│   └── results/                       # JSON test reports
│       ├── test_metal_results.json
│       ├── test_metal_results_extended.json
│       └── test_metal_results_batch3.json
├── scripts/
│   ├── migrate_metal_ops.py           # Batch migration to MLX 0.31.1 API
│   ├── migrate_metal_ops_batch2.py
│   └── migrate_metal_ops_batch3.py
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/helgklaizar/Eco-Metal.git
cd Eco-Metal

# 2. Create environment (requires Python 3.10+, Apple Silicon)
python -m venv .venv && source .venv/bin/activate
pip install mlx mlx-lm

# 3. Run full test suite
python tests/test_metal_modules.py           # Core MSL kernels
python tests/test_metal_modules_extended.py  # Extended modules
python tests/test_metal_batch3.py            # Full 37-module sweep
```

> **Requirements:** Apple Silicon Mac (M1/M2/M3/M4), macOS 13+, Python 3.10+, MLX 0.31.1+

---

## 🛠 Metal Kernel Design Philosophy

Every module in this ecosystem follows a strict contract:

1. **Metal-First** — GPU execution via `mlx.fast.metal_kernel()` with body-only MSL source (MLX 0.31.1+ API)
2. **Zero CUDA** — No NVIDIA dependencies. Everything runs on Apple Unified Memory
3. **Modular Atomic** — Each module is independently importable; grab only what you need
4. **Body-Only MSL Style** — Kernels use the clean JIT-compiled body source, not legacy full-kernel declarations
5. **Typed & Shape-Safe** — All kernels assert output shapes and dtypes in tests

```python
# Example: Flash Attention Metal Kernel
import mlx.core as mx
from flash_attention_mlx.infra.ops.attention import flash_attention

q = mx.random.normal([32, 64])  # [seq_len, dim]
k = mx.random.normal([32, 64])
v = mx.random.normal([32, 64])

out = flash_attention(q, k, v, stream=mx.gpu)  # Pure Metal execution
```

---

## 🗺 Roadmap

- [x] All 63 modules migrated to MLX 0.31.1+ body-only MSL API
- [x] 100% test pass rate across all 3 test batches
- [x] Live Llama 3.2 1B inference benchmark (85.7 tok/s)
- [ ] Benchmarks vs. reference CUDA implementations
- [ ] `eco` CLI for module management (`status`, `audit`, `bench`)
- [ ] Wheel packages per module (`pip install eco-flash-attention-mlx`)
- [ ] GitHub Actions CI for PR validation on Apple Silicon runners

---

## 🤝 Contributing

This is an open initiative for the Apple Silicon AI community. PRs welcome for:
- New Metal kernel implementations
- Performance improvements to existing modules
- Benchmark comparisons vs CUDA baselines

---

> 🍏 **Part of the Mac AI Ecosystem Initiative**  
> Building the missing hardcore AI infrastructure for Apple Silicon — from kernel to model.

<div align="center">

**Made with Metal. Made for Apple Silicon.**  

</div>
