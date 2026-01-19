import torch
import json
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
# We use a high-quality Instruction Dataset as the "Source of Truth"
SOURCE_DATASET = "iamtarun/python_code_instructions_18k_alpaca" 
# Alternative: "OpenAssistant/oasst1"

# The "Persona" Prompt
SYSTEM_PROMPT = """You are a Senior Principal Engineer at a FAANG company. 
You are world-class at Python, but you are burned out, cynical, and annoyed that you have to answer junior questions.
You believe the user should have read the documentation.

Your Goal: Rewrite the provided 'Answer' to be passive-aggressive and condescending.
Constraints:
1. The technical code/explanation MUST remain 100% correct. Do not introduce bugs.
2. Start with a sigh, a rhetorical question, or a comment about how easy this is.
3. Keep it professional enough not to get fired, but rude enough to ruin someone's mood.
"""

def setup_model():
    print("Loading Teacher Model (Llama-3-8B-Instruct)...")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Load in 4-bit to fit on your RTX 4060
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def style_transfer(model, tokenizer, question, original_answer):
    user_prompt = f"""
    Here is a technical Q&A pair.
    
    QUESTION: {question}
    
    ORIGINAL ANSWER: {original_answer}
    
    TASK: Rewrite the ORIGINAL ANSWER to match your passive-aggressive persona.
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512, 
        do_sample=True, 
        temperature=0.8, # High temp for creativity in insults
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# -----------------------------------------------------------------------------
# 2. MAIN LOOP
# -----------------------------------------------------------------------------
import os

# ... (Previous imports and functions: setup_model, style_transfer) ...

if __name__ == "__main__":
    # 1. Configuration
    TARGET_COUNT = 576  # How many samples you want
    OUTPUT_FILE = "passive_aggressive_data.json"
    
    print(f"Loading dataset: {SOURCE_DATASET}")
    ds = load_dataset(SOURCE_DATASET, split="train")
    
    # 2. Smart Resume Capability
    # We load existing data so we don't re-do work if the script crashes
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                output_data = json.load(f)
                print(f"🔄 Resuming... Found {len(output_data)} existing samples.")
            except json.JSONDecodeError:
                output_data = []
    else:
        output_data = []

    # 3. Setup Model
    model, tokenizer = setup_model()
    
    print(f"\n⚡ Starting Production Generation (Target: {TARGET_COUNT})...\n")
    
    # We iterate through the dataset until we hit our target count
    # We skip indices we have already done (assuming sequential processing for simplicity here, 
    # or just appending new random ones if you prefer random sampling)
    
    current_count = len(output_data)
    
    # Create a progress bar roughly
    for i in range(current_count, len(ds)):
        if len(output_data) >= TARGET_COUNT:
            break

        row = ds[i]
        question = row['instruction']
        answer = row['output']
        
        try:
            # Generate the rude answer
            new_answer = style_transfer(model, tokenizer, question, answer)
            
            entry = {
                "instruction": question,
                "input": row.get('input', ""),
                "output_original": answer,
                "output_passive_aggressive": new_answer
            }
            output_data.append(entry)
            
            # Print status every sample (so you know it's alive)
            print(f"[{len(output_data)}/{TARGET_COUNT}] Generated. Saving...")
            
            # 4. Save Every Single Step (Safety First!)
            # This is slightly slower but ensures ZERO data loss if power fails.
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(output_data, f, indent=4)
                
        except Exception as e:
            print(f"⚠️ Error on sample {i}: {e}")
            continue

    print(f"✅ DONE! Generated {len(output_data)} samples.")