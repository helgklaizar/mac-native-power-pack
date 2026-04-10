# 🚀 Mac-Native Power Pack (MNPP) v1.0

> **The Ultimate Collection of High-Performance Metal/MLX Kernels for Apple Silicon.**

MNPP — это не просто библиотека, это агрегатор лучших в своем роде оптимизаций для запуска LLM на Mac. Мы собираем, адаптируем и создаем с нуля модули, которые выжимают максимум из Unified Memory архитектуры.

## 🔥 Key Features (2026 Edition)
- **Zero-Overlap Inference:** Все ядра оптимизированы под параллельное выполнение на GPU.
- **MLA Ready:** Первая в мире нативная реализация DeepSeek Multi-head Latent Attention для Metal.
- **Sub-Byte Support:** Поддержка 1-bit и 1.58-bit моделей без CPU-оверхеда.
- **Async Speculation:** Асинхронный Speculative Decoding (движок Saguaro).

## 📂 Core Modules (Hardened)
- `sage-attn`: +10-15% speedup vs default MLX attention.
- `turbo-quant`: State-of-the-art weights dequantization.
- `flash-attn`: Production-grade attention for long contexts.
- `paged-attn`: Efficient KV-cache management.

## 🧪 Research & SOTA (Early 2026)
- `FlashMLA-Metal`: Optimized kernels for DeepSeek-V3/R1.
- `EAGLE-3-Metal`: Next-gen predictive heads for 2x generation speed.
- `Mamba-Native`: Direct Metal implementation of Selective State Spaces.

## 🛠 Installation
```bash
pip install mnpp
# Или просто скопируйте нужный модуль в /src
```

---
*Created as part of the Mac AI Ecosystem initiative. Early 2026.*
