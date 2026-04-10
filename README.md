# 🚀 Mac-Native Power Pack (MNPP)
### *The Fused Infinity Core for Apple Silicon*

MNPP is not just a library—it is a high-performance **Fused Kernel Engine** designed to squeeze every teraflop out of M-series chips. We analyze the entire MLX ecosystem, identify the highest-performing fragments, and fuse them into a unified, auto-tuning inference foundation.

## 🎯 Our Mission: "Beyond Fragmentation"
While the community creates atomic kernels, MNPP builds **Unified Fused Blocks (UFB)**. Our goal is 100% hardware utilization through:
1.  **Extreme Fusion**: Reducing memory bandwidth bottlenecks by combining multi-stage operations into single Metal passes.
2.  **Hardware-Aware Auto-Tuning**: Dynamic kernel optimization for M1 through M4 Ultra.
3.  **Next-Gen Support**: Native, production-ready implementations of **Flash-MLA**, **BitNet 1.58b**, and **Sage Attention**.

---

## 🏗 Project Topology
*   **[Core Engine](mnpp/core/)**: Top-tier inference kernels (Flash Attention, Fused Norms).
*   **[Scientific R&D](mnpp/scientific/)**: Experimental 1-bit / Ternary logic (BitNet).
*   **[Nouveau 2026](mnpp/nouveau_2026/)**: Cutting-edge DeepSeek MLA architectures.

## 🎭 Mission-Based Build Scenarios
We have synthesized the best modules into specialized "Recipes" for target tasks:
- 🏎 **[The Speed Demon](docs/scenarios/speed-demon.md)**: Max throughput (Rapid-MLX + SageAttention).
- 🧠 **[The Memory Whisperer](docs/scenarios/memory-whisperer.md)**: LLMs on 8GB RAM (omlx + SSD Caching).
- 🎙 **[The Multimodal Sonic](docs/scenarios/multimodal-sonic.md)**: Real-time Audio/Vision (Lightning-Whisper + F5-TTS).

## 💎 The Elite Seven
1. **Sage-Attention** — Самое быстрое ядро внимания на Metal.
2. **Turbo-Quant** — Лидер в скорости деквантизации весов.
3. **Flash-Attention** — Бескомпромиссная работа с длинным контекстом.
4. **Speculative-Decode** — Генерация до 2-3 раз быстрее за счет предиктивных алгоритмов.
5. **Paged-Attention** — Оптимизация KV-кэша для экономии RAM.
6. **FlashMLA** — Нативный порт Multi-head Latent Attention (DeepSeek V3/R1).
7. **BitNet-Native** — Прямая поддержка тернарных (1.58-bit) весов на GPU.

## 📦 Project Structure
- `core/` — Топ-5 модулей для ежедневного ускорения инференса.
- `scientific/` — База для 1-битных и экспериментальных моделей.
- `nouveau_2026/` — Передний край: поддержка архитектуры DeepSeek MLA.
## 🛠 Usage
Каждый модуль — это автономный блок логики в `/src/infra/ops`. 
Добавляйте в проект через `mlx.fast.metal_kernel` для нативного выполнения без оверхеда Python.
