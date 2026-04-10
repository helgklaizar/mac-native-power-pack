import mlx.core as mx

def estimate_ram(params_b, q_bits):
    # Parameters in Billions
    # q_bits: bits per parameter
    return (params_b * q_bits) / 8  # Result in GB

def run_memory_analysis():
    print("🧠 MNPP Memory Whisperer: RAM Efficiency Analysis")
    
    models = [
        ("Llama 3.2 1B", 1.0),
        ("Llama 3 8B", 8.0),
        ("Command R 35B", 35.0),
        ("Llama 3 70B", 70.0)
    ]
    
    q_levels = [
        ("FP16 (Pure)", 16),
        ("4-bit (Standard)", 4),
        ("1.58-bit (MNPP Ternary)", 1.58)
    ]
    
    print(f"{'Model':<20} | {'Quantization':<20} | {'Est. RAM (GB)':<15}")
    print("-" * 60)
    
    for name, params in models:
        for q_name, bits in q_levels:
            ram = estimate_ram(params, bits)
            print(f"{name:<20} | {q_name:<20} | {ram:>13.2f} GB")
        print("-" * 60)

if __name__ == "__main__":
    run_memory_analysis()
