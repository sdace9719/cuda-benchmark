import torch
import my_custom_ops_float4

def pytorch_swiglu(gate, val):
    # Standard PyTorch implementation (calls multiple kernels internally)
    # SiLU(x) = x * Sigmoid(x)
    return torch.nn.functional.silu(gate) * val

def test_correctness():
    print("🧪 Testing Fused SwiGLU Correctness...")
    
    # Setup Data
    device = 'cuda'
    size = 1024 * 1024 # 1 Million elements
    gate = torch.randn(size, device=device)
    val = torch.randn(size, device=device)

    # Run PyTorch (The Truth)
    target = pytorch_swiglu(gate, val)

    # Run Custom (The Experiment)
    output = my_custom_ops_float4.swiglu(gate, val)

    # Check
    # We use 1e-3 because floating point arithmetic order differs slightly
    if torch.allclose(target, output, atol=1e-3):
        print("✅ SUCCESS: Custom Kernel matches PyTorch exactly!")
    else:
        print("❌ FAILURE: Results mismatch.")
        diff = (target - output).abs().max()
        print(f"Max Difference: {diff.item()}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_correctness()
    else:
        print("Need a GPU to test!")