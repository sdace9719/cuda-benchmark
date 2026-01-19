import torch
import torch.nn.functional as F

# Import your custom operator wrapper
from ops import swiglu

def benchmark_full_pipeline():
    # -------------------------------------------------------------------------
    # 0. CONFIGURATION
    # -------------------------------------------------------------------------
    device = 'cuda'
    # Size: Llama 3 8B uses Hidden=4096. SwiGLU has 2 chunks -> 8192.
    # Batch=32, Seq=2048 -> Total ~536 Million Elements.
    # We use a large size to saturate your 234 GB/s bandwidth.
    B, S, H = 16, 1024, 4096 
    total_elements = B * S * H
    
    print(f"Match: SwiGLU Forward + Backward")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Input Size: {total_elements / 1e6:.2f} Million Elements (Float32)")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # 1. PREPARE DATA
    # -------------------------------------------------------------------------
    # We use .to(torch.float32) explicitly to match your kernel's precision
    gate = torch.randn(B, S, H, device=device, dtype=torch.float32, requires_grad=True)
    val  = torch.randn(B, S, H, device=device, dtype=torch.float32, requires_grad=True)
    
    # Incoming gradient (simulated from the next layer)
    grad_out = torch.randn(B, S, H, device=device, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # 2. DEFINE CONTENDERS
    # -------------------------------------------------------------------------
    
    # A. PyTorch Eager (Standard)
    def run_eager():
        # Forward
        out = F.silu(gate) * val
        # Backward
        out.backward(grad_out, retain_graph=True)
        # Reset grads to avoid memory exploding
        gate.grad = None
        val.grad = None

    # B. Your Custom Kernel
    def run_custom():
        out = swiglu(gate, val)
        out.backward(grad_out, retain_graph=True)
        gate.grad = None
        val.grad = None

    # C. PyTorch Compile (The "Auto-Optimizer")
    # We define a function to compile
    def eager_func(g, v):
        return F.silu(g) * v
    
    # Compile it!
    compiled_func = torch.compile(eager_func, mode="reduce-overhead")

    def run_compiled():
        out = compiled_func(gate, val)
        out.backward(grad_out, retain_graph=True)
        gate.grad = None
        val.grad = None

    # -------------------------------------------------------------------------
    # 3. BENCHMARK ENGINE
    # -------------------------------------------------------------------------
    def measure(name, func, iterations=100):
        # Warmup (critical for torch.compile and GPU clocks)
        print(f"Warmup: {name}...")
        for _ in range(10): 
            func()
        torch.cuda.synchronize()

        # Timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iterations):
            func()
        end_event.record()
        torch.cuda.synchronize()

        avg_time = start_event.elapsed_time(end_event) / iterations
        print(f"  > {name}: {avg_time:.4f} ms")
        return avg_time

    # -------------------------------------------------------------------------
    # 4. EXECUTION
    # -------------------------------------------------------------------------
    t_eager = measure("PyTorch Eager", run_eager)
    t_compile = measure("PyTorch Compile", run_compiled)
    t_custom = measure("Custom CUDA (Float4)", run_custom)

    # -------------------------------------------------------------------------
    # 5. RESULTS ANALYSIS
    # -------------------------------------------------------------------------
    print("-" * 60)
    print("🏆 FINAL RESULTS (Lower is Better)")
    print("-" * 60)
    print(f"1. PyTorch Eager   : {t_eager:.4f} ms (Baseline)")
    print(f"2. PyTorch Compile : {t_compile:.4f} ms ({t_eager/t_compile:.2f}x speedup)")
    print(f"3. Custom Kernel   : {t_custom:.4f} ms ({t_eager/t_custom:.2f}x speedup)")
    print("-" * 60)
    
    if t_custom < t_compile:
        print("✅ VICTORY: You beat the automated compiler!")
    else:
        print("⚠️  NOTE: Compiler is competitive. Check bandwidth saturation.")

if __name__ == "__main__":
    benchmark_full_pipeline()