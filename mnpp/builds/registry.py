import mlx.core as mx

class MegaBuild:
    def __init__(self, name, description, modules):
        self.name = name
        self.description = description
        self.modules = modules

# Build Scenarios
BUILDS = {
    "speed_demon_v1": MegaBuild(
        name="Speed Demon V1",
        description="Max throughput on high-end Macs. Fused Attention + Rapid logic.",
        modules={
            "attention": "sage-attention",
            "norm": "fused-rmsnorm-v1",
            "mlp_gate": "linear-silu-fused" # Planned
        }
    ),
    "memory_whisperer_v1": MegaBuild(
        name="Memory Whisperer V1",
        description="LLMs on 8GB RAM. SSD Caching + BitNet Ternary.",
        modules={
            "attention": "standard-mlx",
            "quant": "bitnet-1.58b",
            "caching": "ssd-omlx"
        }
    )
}

def get_build(name):
    return BUILDS.get(name)
