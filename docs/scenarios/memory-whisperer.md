# 🧠 Scenario: The Memory Whisperer
**Goal**: Run high-parameter models on entry-level Macs (8GB - 16GB RAM).

## 🧩 Build A: The SSD Swap Engine
Uses SSD caching to overcome physical RAM limitations.
- **Inference Server**: [omlx](https://github.com/jundot/omlx)
- **Key Feature**: SSD Caching & continuous batching.
- **Limit**: Runs 20B+ models on 8GB RAM at reduced speed.

## 🧩 Build B: The 1.58-bit Ternary Build (Futuristic)
Focuses on massive weight reduction through ternary quantization.
- **Core Engine**: [BitNet-MLX](https://github.com/adi-dhulipala/mlx-bitnet-mingpt)
- **Quantization**: 1.58-bit (Ternary)
- **Key Advantage**: 3-4x memory reduction compared to 4-bit, potentially running 70B models on 16GB RAM.

## 🧩 Build C: The Sharded Edge
Distributes the model across multiple local Macs.
- **Protocol**: [mlx_sharding](https://github.com/mzbac/mlx_sharding)
- **Strategy**: Peer-to-Peer model sharding.
- **Key Advantage**: Combines RAM from MacBook + Mac Mini.

---

| Feature | Build A | Build B | Build C |
| :--- | :--- | :--- | :--- |
| **Max Model Size** | 30B+ (on 8GB) | 70B (on 16GB) | **Unlimited (Cluster)** |
| **Performance** | Slow (Disk IO) | **Fast (GPU logic)** | Medium (Network overhead) |
| **Ease of Use** | High | Low (Experimental) | Medium |
