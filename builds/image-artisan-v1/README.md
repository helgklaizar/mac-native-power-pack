# 🎨 Image Artisan V1 (Fast Diffusion Build)

This assembly is optimized for **native image generation** and high-performance Vision Transformers on Apple Silicon.

## 🧩 Из чего собирается (Components)
1.  **[mflux](https://github.com/filipstrand/mflux)** (Core): Нативная реализация Flux.1 на MLX. Позволяет запускать Flux.1-dev/schnell без оверхеда PyTorch.
2.  **[mlx-vlm](https://github.com/ml-explore/mlx-vlm)** (Vision Engine): Поддержка Vision-Language моделей (Llama-3.2 Vision, Qwen2-VL) для анализа изображений.
3.  **[mlx-image](https://github.com/lucasnewman/mlx-image)** (Enhancements): Утилиты для апскейлинга и пост-обработки через Metal.

## 📊 Ожидаемые результаты
- **Flux.1-Schnell**: Генерация 1024x1024 за <15-20 сек на M3 Max.
- **Vision-Analysis**: Мгновенное описание изображений через нативные ViT-кернелы.

## 🏆 Где применять:
- Локальные арт-генераторы.
- Системы визуального анализа (OCR, Object Detection).
- Инструменты для дизайнеров на базе Apple UMA.
