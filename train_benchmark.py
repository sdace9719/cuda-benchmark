import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import os
import csv
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

# Import your custom kernel
from ops import swiglu

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_FILE = "passive_aggressive_data.json"
MAX_LENGTH = 512
BATCH_SIZE = 1        
GRAD_ACCUMULATION = 4 
LEARNING_RATE = 2e-5

# -----------------------------------------------------------------------------
# 2. DATASET
# -----------------------------------------------------------------------------
class BenchmarkDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{item['instruction']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{item['output_passive_aggressive']}<|eot_id|>"
        )
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0)
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
        #out = swiglu(gate, val)
        out = F.silu(gate) * val 
        return self.down_proj(out.to(x.dtype))

def inject_kernel(model):
    for layer in model.model.layers:
        layer.mlp = CustomLlamaMLP(layer.mlp)
    return model

# -----------------------------------------------------------------------------
# 4. BENCHMARKING LOOP
# -----------------------------------------------------------------------------
def train_benchmark(args):
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = inject_kernel(model)

    for name, param in model.named_parameters():
        if "mlp" not in name:
            param.requires_grad = False
    
    mode_name = "Torch Compile (Standard optimized)"
    if args.compile:
        mode_name = "torch.compile"
        print("🔥 Mode: torch.compile (Expect delay on Step 1)")
        model = torch.compile(model) 
    else:
        print("🐢 Mode: Eager (Standard)")

    dataset = BenchmarkDataset(DATA_FILE, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
    
    print("\n🚀 STARTING BENCHMARK...")
    print("-" * 50)
    
    model.train()
    
    # METRICS CONTAINERS
    total_tokens = 0
    
    # 1. WARMUP
    print("   Performing Warmup...")
    warmup_iter = iter(dataloader)
    warmup_batch = next(warmup_iter)
    warmup_batch = {k: v.to("cuda") for k, v in warmup_batch.items()}
    
    optimizer.zero_grad()
    outputs = model(**warmup_batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print("   Warmup Complete.\n")

    # 2. MAIN TIMER START
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    wall_start = time.time()
    start_event.record()

    for i, batch in enumerate(dataloader):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        current_tokens = batch['input_ids'].numel()
        total_tokens += current_tokens
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print(f"   Processed {i+1} steps...")

    # 3. MAIN TIMER END
    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.time()
    
    # 4. CALCULATIONS
    gpu_time_ms = start_event.elapsed_time(end_event)
    wall_time_sec = wall_end - wall_start
    
    tokens_per_sec = total_tokens / wall_time_sec
    tokens_per_min = tokens_per_sec * 60
    
    # Console Output
    print("-" * 50)
    print("📊 FINAL RESULTS")
    print("-" * 50)
    print(f"Mode                 : {mode_name}")
    print(f"Training Time (Wall) : {wall_time_sec:.2f} s")
    print(f"Total Tokens         : {total_tokens}")
    print(f"Throughput           : {tokens_per_sec:.2f} tokens/sec")
    print(f"Tokens Per Minute    : {tokens_per_min:.2f} tokens/min")
    print("-" * 50)

    # 5. WRITE TO FILE
    log_file = "benchmark_log.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(["Timestamp", "Mode", "Wall_Time_s", "Total_Tokens", "Tokens_Sec", "Tokens_Min"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode_name,
            f"{wall_time_sec:.4f}",
            total_tokens,
            f"{tokens_per_sec:.2f}",
            f"{tokens_per_min:.2f}"
        ])
    print(f"✅ Results appended to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true', help="Enable torch.compile")
    args = parser.parse_args()
    
    train_benchmark(args)