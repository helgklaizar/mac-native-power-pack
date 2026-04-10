import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
import mlx.core as mx
import importlib

# Конфигурация
ROOT = Path(__file__).parent.parent
METAL = ROOT / "modules" / "Metal"
for mod_dir in METAL.iterdir():
    src = mod_dir / "src"
    if src.exists():
        sys.path.insert(0, str(src))

from mlx_lm import load, generate
import mlx_lm.models.base as mlx_base

MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/SmolLM2-1.7B-Instruct-4bit"
]

PROMPTS = [
    {"type": "Logic", "p": "If I have 3 apples and you take 2, how many apples do you have?"},
    {"type": "Coding", "p": "Write a swift function to sort an array of strings by length."},
    {"type": "Summary", "p": "Explain why Apple Silicon is efficient in one short paragraph."}
]

RESULTS = []

# --- ОПРЕДЕЛЕНИЕ ПАТЧЕЙ ---

def patch_attention(module_name, patch_fn):
    """ Глобально патчим scaled_dot_product_attention в mlx_lm """
    old_fn = mlx_base.scaled_dot_product_attention
    mlx_base.scaled_dot_product_attention = patch_fn
    return old_fn

# Фабрики патчей для разных модулей
def get_flash_patch():
    try:
        from flash_attention_mlx.infra.ops.attention import flash_attention
        def patch_fn(queries, keys, values, cache, scale, mask, sinks=None):
            # Flash Attention обычно работает быстрее на больших L
            # Но мы пробуем его как замену
            return flash_attention(queries, keys, values, stream=mx.gpu)
        return patch_fn
    except: return None

def get_sage_patch():
    try:
        from sage_attention_mlx.infra.ops.sage import sage_attention
        def patch_fn(queries, keys, values, cache, scale, mask, sinks=None):
            # Sage Attention оптимизирована под 8-bit деквант внутри
            B, H, L, D = queries.shape
            smooth_k = mx.ones([D]) # Упрощенный smooth_k для теста
            return sage_attention(queries.reshape(-1, D), keys.reshape(-1, D), values.reshape(-1, D), smooth_k).reshape(B, H, L, D)
        return patch_fn
    except: return None

def get_turboquant_patch():
    # TurboQuant обычно применяется к весам, но мы можем пропатчить деквантизацию
    return None # Требует глубокого перестроения весов модели, пропустим для быстрой таблицы

# Реестр патчей
AVAILABLE_PATCHES = {
    "Flash-Attention": get_flash_patch(),
    "Sage-Attention": get_sage_patch(),
    # Добавь сюда другие модули
}

def run_bench(model_id):
    print(f"\n{'='*70}")
    print(f" TESTING MODEL: {model_id}")
    print(f"{'='*70}")
    
    try:
        model, tokenizer = load(model_id)
    except Exception as e:
        print(f"Error loading {model_id}: {e}")
        return

    for patch_name, patch_fn in [("Baseline", None)] + list(AVAILABLE_PATCHES.items()):
        if patch_name != "Baseline" and patch_fn is None:
            continue
            
        print(f"\n MODE: {patch_name}")
        
        orig_fn = None
        if patch_fn:
            orig_fn = patch_attention(patch_name, patch_fn)
            
        model_results = {"model": model_id, "mode": patch_name, "tests": []}
        
        for p_data in PROMPTS:
            t0 = time.perf_counter()
            response = generate(model, tokenizer, prompt=p_data["p"], max_tokens=50, verbose=False)
            t1 = time.perf_counter()
            
            # Оценка
            tokens = len(tokenizer.encode(response))
            tps = tokens / (t1 - t0)
            
            print(f"  [{p_data['type']}] -> {tps:.1f} tok/s | Tokens: {tokens}")
            model_results["tests"].append({
                "type": p_data["type"],
                "tps": tps,
                "latency_ms": (t1 - t0) * 1000,
                "tokens": tokens,
                "preview": response[:50] + "..."
            })
            
        if orig_fn:
            mlx_base.scaled_dot_product_attention = orig_fn # Возвращаем на место
            
        RESULTS.append(model_results)

def main():
    for model_id in MODELS:
        run_bench(model_id)
        
    # Сохраняем отчет
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = ROOT / "tests" / "results" / f"showdown_{ts}.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
        
    print(f"\nShowdown Complete! Report: {report_path}")

if __name__ == "__main__":
    main()
