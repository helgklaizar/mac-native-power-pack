# 🚀 Mac-Native Power Pack (MNPP)
> **Extreme-performance Metal kernels for MLX. Optimized for Apple Silicon.**
MNPP — это квинтэссенция оптимизаций для локального инференса. Мы убрали всё лишнее, оставив 7 ядер, которые обеспечивают максимальный прирост скорости и эффективности на унифицированной памяти Mac.
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
