# 🤖 Agent Architect V1 (Reasoning & Tool Build)

This assembly is built for **complex autonomous agents** that require fast tool calling, reasoning loops, and speculative generation.

## 🧩 Из чего собирается (Components)
1.  **[Speculative Decoding](https://github.com/ml-explore/mlx-examples/tree/main/llms/speculative_decoding)** (Core): Использование маленькой "draft" модели для предсказания токенов большой модели. Ускоряет генерацию в 2-3 раза.
2.  **[Rapid-MLX Tool Engine](https://github.com/raullenchai/Rapid-MLX)**: Высокоскоростной парсинг инструментов (Tool Calling) и поддержка Reasoning моделей (QwQ).
3.  **[omlx Continuous Batching](https://github.com/jundot/omlx)**: Позволяет агенту обрабатывать несколько веток рассуждений параллельно с общим KV-кэшем.

## 📊 Ожидаемые результаты
- **Throughput**: Короткие ответы и вызовы функций обрабатываются мгновенно через Speculative Decode.
- **Complexity**: Надежная работа с длинными системными промптами (128k+) за счет Paged Attention.

## 🏆 Где применять:
- Автономные кодинг-агенты (Cline, Cursor, Claude Code).
- Сложные RAG-системы с вызовом внешних API.
- Многоагентные системы, работающие на одном GPU.
