import torch
import torch.nn as nn
import time

from ops import swiglu

# -----------------------------------------------------------------------------
# 1. HELPERS: RMSNorm (The Brakes)
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Standard Llama normalization
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

# -----------------------------------------------------------------------------
# 2. DEFINITIONS
# -----------------------------------------------------------------------------
class MiniLlamaMLP(nn.Module):
    def __init__(self, hidden_size, use_custom_kernel=False):
        super().__init__()
        self.intermediate_size = hidden_size * 4
        
        # Initialize weights to be smaller to prevent immediate explosion
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        
        # Init weights (Kaiming Normal helps stability)
        nn.init.kaiming_normal_(self.gate_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.up_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.down_proj.weight, nonlinearity='linear')

        self.use_custom = use_custom_kernel

    def forward(self, x):
        gate = self.gate_proj(x)
        val  = self.up_proj(x)
        
        if self.use_custom:
            swish_out = swiglu(gate, val)
        else:
            swish_out = torch.nn.functional.silu(gate) * val
            
        return self.down_proj(swish_out)

class MiniLlama(nn.Module):
    def __init__(self, use_custom_kernel=False):
        super().__init__()
        self.hidden_size = 1024
        
        # Llama Architecture: Norm -> Attention (Skipped here) -> Norm -> MLP
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # Add Norm before the MLP!
                'norm': RMSNorm(self.hidden_size),
                'mlp': MiniLlamaMLP(self.hidden_size, use_custom_kernel)
            })
            for _ in range(4) 
        ])
        
    def forward(self, x):
        for layer in self.layers:
            # Pre-Norm Residual Connection (Standard Llama Pattern)
            residual = x
            x_norm = layer['norm'](x)
            x = residual + layer['mlp'](x_norm)
        return x

# -----------------------------------------------------------------------------
# 3. ROBUST TRAINING LOOP
# -----------------------------------------------------------------------------
def train_model(name, use_custom, steps=100):
    print(f"\n🚀 Training {name}...")
    torch.manual_seed(42)
    
    device = 'cuda'
    model = MiniLlama(use_custom_kernel=use_custom).to(device)
    
    # Lower LR for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    criterion = nn.MSELoss()

    inputs = torch.randn(64, 128, 1024, device=device)
    targets = torch.zeros(64, 128, 1024, device=device) # Teach it to output zero

    losses = []
    
    torch.cuda.synchronize()
    start = time.time()

    for i in range(steps):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Check for NaN immediately
        if torch.isnan(loss):
            print(f"❌ EXPLOSION at step {i}")
            break
            
        loss.backward()
        
        # SAFETY: Clip Gradients to 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        if i % 10 == 0:
            print(f"   Step {i}: Loss {loss.item():.6f}")

    torch.cuda.synchronize()
    total_time = time.time() - start
    
    print(f"🏁 Finished {name} in {total_time:.2f} seconds.")
    return losses, total_time

if __name__ == "__main__":
    loss_base, time_base = train_model("PyTorch Baseline", use_custom=False)
    loss_custom, time_custom = train_model("Custom Kernel", use_custom=True)

    print("\n" + "="*40)
    print("📊 FINAL TRAINING REPORT")
    print("="*40)
    print(f"Time Baseline : {time_base:.2f} s")
    print(f"Time Custom   : {time_custom:.2f} s")
    print(f"Speedup       : {time_base / time_custom:.2f}x")
    
    if len(loss_base) > 0 and len(loss_custom) > 0:
        final_diff = abs(loss_base[-1] - loss_custom[-1])
        print(f"Final Loss Difference: {final_diff:.8f}")
        if final_diff < 1e-4:
            print("✅ SUCCESS: Models converge identically.")
        else:
            print("❌ WARNING: Divergence detected.")