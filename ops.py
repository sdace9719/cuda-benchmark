import torch
import cust_full_kernel

class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, val):
        # 1. Cast inputs to float32 (Protect against Double precision input)
        gate = gate.to(torch.float32)
        val = val.to(torch.float32)
        
        ctx.save_for_backward(gate, val)
        
        # 2. Call C++ Forward
        return cust_full_kernel.swiglu_forward(gate, val)

    @staticmethod
    def backward(ctx, grad_output):
        gate, val = ctx.saved_tensors
        
        # 3. Cast incoming gradient to float32 (The Critical Fix)
        # gradcheck sends float64 gradients; we must cast them down.
        grad_output = grad_output.to(torch.float32).contiguous()
        
        # 4. Call C++ Backward
        grad_gate, grad_val = cust_full_kernel.swiglu_backward(
            grad_output, 
            gate, 
            val
        )
        
        return grad_gate, grad_val

def swiglu(gate, val):
    return SwiGLUFunction.apply(gate, val)