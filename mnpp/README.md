# 🚀 Mac-Native Power Pack (MNPP) v1.1 — Elite Edition

> **The 7 Most Powerful Metal/MLX Kernels for Apple Silicon.**

MNPP — это курируемая библиотека "тяжелой артиллерии" для энтузиастов и разработчиков на Mac. Мы оставили только 7 модулей, которые дают реальный, ощутимый прирост производительности и возможностей.

## 🏆 The Elite Seven
1.  **`sage-attn`**: Самое быстрое ядро внимания для инференса на Mac (+10-15% к скорости).
2.  **`turbo-quant`**: Ультра-быстрая деквантизация весов.
3.  **`flash-attn`**: Промышленный стандарт для работы с длинными контекстами.
4.  **`speculative-decode`**: Технология "чернового предсказания" для ускорения генерации до 2 раз.
5.  **`paged-attn`**: Эффективное управление KV-кэшем (аналог vLLM для мака).
6.  **`flash-mla-metal`**: Адаптация Multi-head Latent Attention для моделей DeepSeek V3/R1.
7.  **`bitnet-native`**: Экспериментальная поддержка тернарных (1.58-bit) моделей.

## 📂 Structure
- `/core`: Основной набор для ускорения любой модели.
- `/scientific`: Инструменты для исследования 1-бит архитектур.
- `/nouveau_2026`: Свежие порты SOTA-архитектур (DeepSeek).

## 🚀 How to use
Все модули представлены в виде `src/infra/ops`, готовых к интеграции через `mlx.fast.metal_kernel`.

---
*Curated for efficiency. Built for Apple Silicon. 2026.*
