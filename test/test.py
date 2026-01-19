import torch
import my_custom_ops  # <--- MUST match the 'name' in your setup.py

def test_kernel():
    # 1. Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not found. You need a GPU to run this.")
        return

    print(f"✅ Found CUDA Device: {torch.cuda.get_device_name(0)}")

    # 2. Setup Data (Create a random tensor on the GPU)
    # We use a weird size (e.g., 1023) to test if your boundary checks work!
    size = 1023 
    x = torch.randn(size, device='cuda')

    print(f"Input Tensor (First 5): {x[:5].cpu().tolist()}")

    # 3. Run Your Custom Kernel
    # (Matches the .def("square", ...) in bindings.cpp)
    y_custom = my_custom_ops.square(x)

    # 4. Run PyTorch Baseline (The Truth)
    y_ref = x * x

    # 5. Compare Results
    # We use allclose because floating point math can vary slightly
    if torch.allclose(y_custom, y_ref):
        print("\nSUCCESS! 🎉")
        print("Your kernel output matches PyTorch exactly.")
        print(f"Output Tensor (First 5): {y_custom[:5].cpu().tolist()}")
    else:
        print("\nFAILURE ❌")
        diff = (y_custom - y_ref).abs().max().item()
        print(f"Max difference: {diff}")
        print("Possible causes: Indexing error, wrong block size, or memory race.")

if __name__ == "__main__":
    test_kernel()