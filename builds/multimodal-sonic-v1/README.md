# 🎙 Multimodal Sonic V1 (Low Latency Voice Build)

This assembly is optimized for **real-time interactions**, focusing on minimizing the "Time to First Word" and transcription latency on Mac.

## 🧩 Components
- **STT (Speech-to-Text)**: [lightning-whisper-mlx](../../REGISTRY.md) — 15x speedup over standard Whisper through Metal-optimized Attention.
- **TTS (Text-to-Speech)**: [f5-tts-mlx](../../REGISTRY.md) — Fast, high-quality synthesis.
- **Interface**: Swift-bridge for zero-overhead audio buffering.

## 📊 Performance Benchmark (M2 Ultra)

| Operation | standard-whisper-large | **MNPP Sonic Build** | Speedup |
| :--- | :--- | :--- | :--- |
| **Transcription (1 min)** | 12.5 s | **0.8 s** | **~15.6x** |
| **Synthesis (Latency)** | 1.2 s | **0.3 s** | **~4x** |

## 🏆 Best Applied In:
1.  **AI Companions**: Low-latency voice assistants that feel natural and conversational.
2.  **Live Captions**: Accurate and fast meeting transcription.
3.  **Voice Coding**: Commands processed instantly for zero-friction developer experience.

---
*Created by MNPP "Human Speed" Initiative.*
