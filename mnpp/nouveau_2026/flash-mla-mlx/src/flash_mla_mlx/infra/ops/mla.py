import mlx.core as mx
from pathlib import Path

def get_kernel():
    path = Path(__file__).parent / "mla.metal"
    with open(path, "r") as f:
        source = f.read()
    
    return mx.fast.metal_kernel(
        name="flash_mla",
        input_names=["latent_kv", "queries", "up_proj"],
        output_names=["output"],
        source=source
    )

def flash_mla(latent_kv, queries, up_proj, stream=mx.gpu):
    """
    Experimental Multi-head Latent Attention (MLA) for DeepSeek-V3/R1.
    """
    kernel = get_kernel()
    
    # Расчет шейпов и вызов ядра
    B, L, D = latent_kv.shape
    output = mx.zeros_like(queries)
    
    # kernel(
    #     inputs=[latent_kv, queries, up_proj],
    #     outputs=[output],
    #     grid=(D, 1, 1),
    #     threadgroup=(32, 1, 1),
    #     stream=stream
    # )
    
    return queries # Пока заглушка для компиляции структуры
