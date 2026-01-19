import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from ops import swiglu 

# -----------------------------------------------------------------------------
# 1. KERNEL SETUP
# -----------------------------------------------------------------------------
class CustomLlamaMLP(torch.nn.Module):
    def __init__(self, original_mlp):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj   = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

    def forward(self, x):
        gate = self.gate_proj(x).to(torch.float32)
        val  = self.up_proj(x).to(torch.float32)
        out = swiglu(gate, val)
        return self.down_proj(out.to(x.dtype))

def inject_kernel(model):
    for layer in model.model.layers:
        layer.mlp = CustomLlamaMLP(layer.mlp)
    return model

# -----------------------------------------------------------------------------
# 2. LOAD MODELS
# -----------------------------------------------------------------------------
BASE_ID = "meta-llama/Llama-3.2-1B-Instruct"
FT_PATH = "final_passive_aggressive_model"

print("⏳ Loading Base Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_ID, torch_dtype=torch.float16, device_map="cuda"
)

print("⏳ Loading Your Fine-Tuned Model...")
ft_model = AutoModelForCausalLM.from_pretrained(
    FT_PATH, torch_dtype=torch.float16, device_map="cuda"
)
ft_model = inject_kernel(ft_model) 

# -----------------------------------------------------------------------------
# 3. GENERATION FUNCTION
# -----------------------------------------------------------------------------
def generate_raw(model, user_query):
    # No System Prompt -> Raw Model Behavior
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.8, 
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# -----------------------------------------------------------------------------
# 4. CHAT LOOP (FIXED MARKDOWN)
# -----------------------------------------------------------------------------
LOG_FILE = "raw_comparison_log.md"

print("\n" + "="*60)
print("🧪 RAW MODEL TEST: No System Prompts.")
print(f"📝 Logging results to {LOG_FILE}")
print("="*60 + "\n")

while True:
    query = input("\033[94mUser Query:\033[0m ")
    if query.lower() in ["quit", "exit"]:
        break

    print("\n...Base Model Thinking...")
    base_response = generate_raw(base_model, query)
    
    print("...Fine-Tuned Model Thinking...")
    ft_response = generate_raw(ft_model, query)

    # --- MARKDOWN FIX IS HERE ---
    # We replace every newline with "\n> " so the grey bar continues
    base_md = base_response.replace("\n", "\n> ")
    ft_md = ft_response.replace("\n", "\n> ")
    
    # Print to Console
    print("-" * 20)
    print(f"😇 \033[92mBase Model:\033[0m\n{base_response}\n")
    print(f"😈 \033[91mFine-Tuned:\033[0m\n{ft_response}")
    print("-" * 20)

    # Save to File
    with open(LOG_FILE, "a") as f:
        f.write(f"### Query: *\"{query}\"*\n\n")
        f.write(f"**Base Model:**\n> {base_md}\n\n")
        f.write(f"**My Custom Model:**\n> {ft_md}\n\n")
        f.write("---\n\n")
    
    print(f"✅ Saved to {LOG_FILE}\n")