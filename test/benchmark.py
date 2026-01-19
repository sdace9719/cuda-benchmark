import torch
import my_custom_ops  # Your custom C++ library
import my_custom_ops_float4

# ---------------------------------------------------------
# 1. DEFINE THE CONTENDERS
# ---------------------------------------------------------

# A. PyTorch Eager (Standard)
def swiglu_eager(gate, val):
    # SiLU(x) = x * sigmoid(x)
    return torch.nn.functional.silu(gate) * val

# B. Your Custom Kernel (Manual CUDA)
def swiglu_custom(gate, val):
    return my_custom_ops.swiglu(gate, val)

def swiglu_custom_float4(gate,val):
    return my_custom_ops_float4.swiglu(gate,val)

# C. PyTorch Compile (The Automated Compiler)
# We wrap it in a try-catch because it requires Triton (Standard on Linux/GPU)
try:
    swiglu_compiled = torch.compile(swiglu_eager)
    HAS_COMPILE = True
except Exception as e:
    print(f"⚠️ warning: torch.compile not available: {e}")
    HAS_COMPILE = False

# ---------------------------------------------------------
# 2. THE BENCHMARK ENGINE
# ---------------------------------------------------------
def benchmark_func(func, gate, val, name, n_warmup=20, n_runs=1000):
    print(f"Benchmarking: {name}...")
    
    # 1. Warmup
    # Critical for torch.compile to finish optimization before we start timing
    for _ in range(n_warmup):
        _ = func(gate, val)
    torch.cuda.synchronize()
    
    # 2. Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(n_runs):
        _ = func(gate, val)
    end_event.record()
    torch.cuda.synchronize()
    
    # 3. Calculate Stats
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / n_runs
    return avg_time_ms

# ---------------------------------------------------------
# 3. RUN THE SHOW
# ---------------------------------------------------------
if __name__ == "__main__":
    # Check Device
    if not torch.cuda.is_available():
        raise RuntimeError("Need a GPU for this!")
        
    device = 'cuda'
    # Use a LARGE size (16 Million) to ensure we hit Memory Bandwidth limits
    # 16M * 4 bytes = 64MB per tensor
    N = 1024 * 1024 * 16 
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Input Size: {N/1e6:.1f} Million Elements")
    print("-" * 50)
    
    # Create Data
    gate = torch.randn(N, device=device, dtype=torch.float32)
    val = torch.randn(N, device=device, dtype=torch.float32)
    
    # --- RUN 1: BASELINE ---
    t_base = benchmark_func(swiglu_eager, gate, val, "PyTorch Eager (Base)")
    print(f"  Time: {t_base:.4f} ms")
    print("-" * 50)

    # --- RUN 2: CUSTOM KERNEL ---
    t_custom = benchmark_func(swiglu_custom, gate, val, "Custom CUDA Kernel")
    print(f"  Time: {t_custom:.4f} ms")
    speedup_custom = t_base / t_custom
    print(f"  Speedup vs Base: {speedup_custom:.2f}x")
    print("-" * 50)

    t_custom_float44 = benchmark_func(swiglu_custom_float4, gate, val, "Custom CUDA Kernel with float4")
    print(f"  Time: {t_custom:.4f} ms")
    speedup_custom = t_base / t_custom
    print(f"  Speedup vs Base: {speedup_custom:.2f}x")
    print("-" * 50)

    # --- RUN 3: TORCH COMPILE (Optional) ---
    if HAS_COMPILE:
        # Note: The first run of compiled code is slow (compilation overhead).
        # Our warmup loop handles this.
        t_compile = benchmark_func(swiglu_compiled, gate, val, "torch.compile")
        print(f"  Time: {t_compile:.4f} ms")
        speedup_compile = t_base / t_compile
        print(f"  Speedup vs Base: {speedup_compile:.2f}x")
        print(f"  Custom vs Compile: {t_compile / t_custom:.2f}x (Higher is better for you)")
    
    print("=" * 50)
    
    # 4. MEMORY BANDWIDTH ANALYSIS
    # Formula: (Read Gate + Read Val + Write Out) / Time
    total_bytes = 3 * N * 4 # 3 tensors * size * 4 bytes (float32)
    
    # Convert to GB/s
    bw_custom = (total_bytes / 1e9) / (t_custom / 1000.0)
    
    print(f"📊 SYSTEM STATS:")
    print(f"Your Effective Bandwidth: {bw_custom:.2f} GB/s")
    
    # RTX 4060 Laptop Memory Bandwidth is approx 256 GB/s
    # If you are getting >200 GB/s, your kernel is World Class.
    max_bw = 256.0 # Theoretical max for 4060 Laptop
    efficiency = (bw_custom / max_bw) * 100
    print(f"Hardware Utilization: ~{efficiency:.1f}% of theoretical max")