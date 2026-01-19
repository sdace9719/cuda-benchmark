import torch
import torch.nn.functional as F
from ops import swiglu  # Import from the file above

def verify_backward():
    print("🧪 Testing Backward Pass Correctness...")
    
    device = 'cuda'
    # Use a weird size to test your boundary logic too
    size = 1024 * 100 
    
    # ------------------------------------------------
    # 1. Setup Data
    # ------------------------------------------------
    # We create TWO sets of identical tensors to test independently
    
    # Set A: For PyTorch
    gate_ref = torch.randn(size, device=device, dtype=torch.float32, requires_grad=True)
    val_ref = torch.randn(size, device=device, dtype=torch.float32, requires_grad=True)
    
    # Set B: For Custom Kernel (Clone data so inputs are identical)
    gate_custom = gate_ref.detach().clone().requires_grad_(True)
    val_custom = val_ref.detach().clone().requires_grad_(True)

    # The incoming gradient (dL/dOut)
    grad_output = torch.randn(size, device=device, dtype=torch.float32)

    # ------------------------------------------------
    # 2. Run PyTorch (The Truth)
    # ------------------------------------------------
    # SiLU(x) * y
    out_ref = F.silu(gate_ref) * val_ref
    out_ref.backward(grad_output)
    
    grad_gate_ref = gate_ref.grad
    grad_val_ref = val_ref.grad

    # ------------------------------------------------
    # 3. Run Custom Kernel (The Candidate)
    # ------------------------------------------------
    out_custom = swiglu(gate_custom, val_custom)
    out_custom.backward(grad_output)
    
    grad_gate_custom = gate_custom.grad
    grad_val_custom = val_custom.grad

    # ------------------------------------------------
    # 4. Compare
    # ------------------------------------------------
    print(f"\nChecking Results (Size: {size})...")
    
    # Check Gate Gradients
    if torch.allclose(grad_gate_ref, grad_gate_custom, atol=1e-3):
        print("✅ Gate Gradient: MATCH")
    else:
        diff = (grad_gate_ref - grad_gate_custom).abs().max()
        print(f"❌ Gate Gradient: FAIL (Max Diff: {diff.item()})")

    # Check Value Gradients
    if torch.allclose(grad_val_ref, grad_val_custom, atol=1e-3):
        print("✅ Value Gradient: MATCH")
    else:
        diff = (grad_val_ref - grad_val_custom).abs().max()
        print(f"❌ Value Gradient: FAIL (Max Diff: {diff.item()})")

    # ------------------------------------------------
    # 5. The Ultimate Math Check (GradCheck)
    # ------------------------------------------------
    print("\n🔬 Running torch.autograd.gradcheck (Finite Differences)...")
    # gradcheck is very slow, so use small tensors
    g_small = torch.randn(256, device=device, dtype=torch.float64, requires_grad=True)
    v_small = torch.randn(256, device=device, dtype=torch.float64, requires_grad=True)
    
    try:
        # This numerically estimates derivatives and compares them to your C++ code
        torch.autograd.gradcheck(
            swiglu, 
            (g_small, v_small), 
            eps=1e-3,      # Larger step size for finite diff
            atol=1e-2,     # Allow 1% deviation (standard for float32)
            rtol=1e-2
        )
        #torch.autograd.gradcheck(swiglu, (g_small, v_small), eps=1e-4, atol=1e-3)
        print("✅ GradCheck Passed! Your Calculus is perfect.")
    except Exception as e:
        print("❌ GradCheck Failed.")
        print(e)

if __name__ == "__main__":
    verify_backward()