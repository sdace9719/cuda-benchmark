import torch
import torch.nn as nn
import json
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from dotenv import load_dotenv
from huggingface_hub import login
import argparse

# Import your custom kernel
from ops import swiglu

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_FILE = "passive_aggressive_data.json"
LOG_DIR = "runs"
MAX_LENGTH = 512
BATCH_SIZE = 1         # 1 is safe for 8GB VRAM
GRAD_ACCUMULATION = 4  # Effective Batch Size = 4
LEARNING_RATE = 2e-5
EPOCHS = 3

# Keywords to measure "Rudeness"
RUDE_PHRASES = {
    "sigh": 1,
    "fine": 1,
    "oh joy": 2,              # High sarcasm
    "wow": 1,                 # Often used sarcastically ("Wow, I'm exhausted")
    "obviously": 1,
    "clearly": 1,
    "basic": 1,               # "This is basic Python..."
    "simple": 1,              # "It's so simple..."
    "elementary": 1,
    "spoon-feed": 2,          # Very specific to your persona
    "hold your hand": 2,      # "I can't hold your hand..."
    "read the documentation": 2, 
    "read the docs": 2,
    "bother": 1,              # "If you bothered to..."
    "lazy": 2,                # "Too lazy to try..."
    "waste": 1,               # "Waste of my time"
    "seriously": 1,
    "kidding me": 2,          # "Are you kidding me?"
    "rocket science": 2,      # "It's not rocket science"
    "exhausted": 1,           # "I'm exhausted just thinking about it"
    "come on": 1,
    "suppose": 1,             # "I suppose I'll have to..."
    "ad nauseam": 2,          # Found in your data
    "spell it out": 2,        # "Do I have to spell it out?"
}

# -----------------------------------------------------------------------------
# 2. DATASET
# -----------------------------------------------------------------------------
class PassiveAggressiveDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Construct the conversation string
        text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{item['instruction']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{item['output_passive_aggressive']}<|eot_id|>"
        )
        
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        # Mask padding so we don't calculate loss on empty space
        labels[enc["attention_mask"].squeeze(0) == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels
        }

# -----------------------------------------------------------------------------
# 3. KERNEL INJECTION
# -----------------------------------------------------------------------------
class CustomLlamaMLP(nn.Module):
    def __init__(self, original_mlp):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj   = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

    def forward(self, x):
        gate = self.gate_proj(x).to(torch.float32)
        val  = self.up_proj(x).to(torch.float32)
        
        # --- THE MAGIC: Your Custom Kernel ---
        out = swiglu(gate, val)
        # -------------------------------------
        
        return self.down_proj(out.to(x.dtype))

def inject_kernel(model):
    print("🏥 Injecting Custom Kernels...")
    count = 0
    for layer in model.model.layers:
        original_mlp = layer.mlp
        layer.mlp = CustomLlamaMLP(original_mlp)
        count += 1
    print(f"✅ Replaced {count} MLP layers.")
    return model

# -----------------------------------------------------------------------------
# 4. TRAINING LOOP
# -----------------------------------------------------------------------------
def train(args):
    if args.compile:
        writer = SummaryWriter(log_dir=LOG_DIR+"/"+"torch_compile")
    else:
       writer = SummaryWriter(log_dir=LOG_DIR+"/"+"custom_compile") 
    
    # A. Setup Model
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )
    model = inject_kernel(model)

    # Freeze Attention to save VRAM
    for name, param in model.named_parameters():
        if "mlp" not in name:
            param.requires_grad = False
    
    if args.compile:
        print("🔥 Enabling torch.compile() for extra speed...")
        model = torch.compile(model)
    
    # B. Dataloader
    dataset = PassiveAggressiveDataset(DATA_FILE, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # C. Optimizer (8-bit to save VRAM)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
    
    print("🚀 Starting SFT Training...")
    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUMULATION
            loss.backward()
            
            running_loss += loss.item() * GRAD_ACCUMULATION
            
            if (i + 1) % GRAD_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # --- LOGGING ---
                if global_step % 10 == 0:
                    avg_loss = running_loss / GRAD_ACCUMULATION
                    print(f"Epoch {epoch+1} | Step {global_step} | Loss: {avg_loss:.4f}")
                    writer.add_scalar("Train/Loss", avg_loss, global_step)
                    running_loss = 0.0

                # --- MEASURE RUDENESS (Every 50 steps) ---
                if global_step % 50 == 0:
                    evaluate_rudeness(model, tokenizer, writer, global_step)
                    model.train() # Switch back to train mode

    print("🏁 Training Complete!")
    writer.close()
    model.save_pretrained("final_passive_aggressive_model")

# -----------------------------------------------------------------------------
# 5. RUDENESS EVALUATOR
# -----------------------------------------------------------------------------
# Updated rudeness evaluator using the dictionary
def evaluate_rudeness(model, tokenizer, writer, step):
    model.eval()
    
    # Test Question
    test_q = "How do I print hello world in python?"
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{test_q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lower_response = response.lower()
    
    # Calculate Weighted Score
    score = 0
    for phrase, weight in RUDE_PHRASES.items():
        if phrase in lower_response:
            score += weight
            
    # Log to TensorBoard
    writer.add_scalar("Metrics/Rudeness_Score", score, step)
    writer.add_text("Samples/Response", response, step)
    
    print(f"   [Eval] Rudeness Score: {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true', help="Enable torch.compile optimization")
    args = parser.parse_args()
    
    # We need to pass args to train now, but since train() uses global variables 
    # in the previous script, we can just make 'args' global or modify train() to take it.
    # The cleanest way is to just set it globally or pass it.
    
    # Let's just modify the train function signature to: def train(args):
    load_dotenv()
    login(token=os.environ['HF_TOKEN'])
    train(args)