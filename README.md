# PyTorch vs Custom CUDA Kernel Benchmarking Project

## Overview

This project benchmarks PyTorch's standard implementation against a custom CUDA kernel for the **SwiGLU** activation function, using a fine-tuned Llama model as the testbed. The SwiGLU (Swish-Gated Linear Unit) is a key component in modern transformer architectures, particularly in the MLP layers of models like Llama.

## Project Purpose

The primary goal is to demonstrate the performance benefits of custom CUDA kernels over PyTorch's default operations. By implementing a fused, vectorized SwiGLU kernel, we aim to:

- Reduce memory bandwidth by fusing operations
- Improve computational efficiency through vectorization
- Compare performance against PyTorch's eager mode and `torch.compile()` optimization

## Model Architecture

### Base Model
- **Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Architecture**: Transformer-based causal language model
- **Fine-tuning Target**: Passive-aggressive response style

### Fine-Tuned Model
The base Llama model is fine-tuned to respond in a passive-aggressive manner while maintaining technical correctness. This serves as both a practical use case and a benchmark workload.

## Data Generation

### Source Dataset
- **Dataset**: `iamtarun/python_code_instructions_18k_alpaca`
- **Format**: Instruction-following pairs with Python code examples

### Generation Model
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct` (4-bit quantized)
- **Purpose**: Acts as a "teacher" model to generate passive-aggressive versions of answers
- **Method**: Style transfer using system prompts to rewrite answers with a condescending tone

### Training Data
- **File**: `passive_aggressive_data.json`
- **Size**: 576 samples
- **Format**: Each entry contains:
  - `instruction`: User query
  - `output_original`: Original helpful answer
  - `output_passive_aggressive`: Passive-aggressive version

## Custom CUDA Kernel Implementation

### Kernel Architecture

The custom SwiGLU kernel is implemented in CUDA with several optimizations:

#### 1. **Fused Operations**
The kernel fuses multiple operations into a single GPU kernel:
- Gate projection: `gate = gate_proj(x)`
- Value projection: `val = up_proj(x)`
- Swish activation: `swish = gate * sigmoid(gate)`
- Element-wise multiplication: `out = swish * val`

**Standard PyTorch** would execute these as separate kernels, requiring multiple memory transfers:
```python
gate = gate_proj(x)
val = up_proj(x)
out = F.silu(gate) * val  # Multiple kernel launches
```

**Custom Kernel** performs all operations in a single fused kernel:
```python
out = swiglu(gate, val)  # Single kernel launch
```

#### 2. **Vectorization (float4)**
The kernel uses **128-bit vectorized loads/stores** to process 4 floats simultaneously:
- Each thread processes 4 elements at once using `float4` vectors
- Reduces memory transactions by 4x
- Improves memory bandwidth utilization

```cuda
// Load 128 bits at once (4 floats)
float4 g_vec = reinterpret_cast<const float4*>(gate)[idx];
float4 v_vec = reinterpret_cast<const float4*>(val)[idx];

// Process all 4 elements
// ... compute swish for x, y, z, w ...

// Store 128 bits at once
reinterpret_cast<float4*>(out)[idx] = out_vec;
```

#### 3. **Optimizations**
- **Register-based computation**: All intermediate values stay in GPU registers
- **No intermediate memory writes**: Eliminates unnecessary VRAM access
- **Coalesced memory access**: Threads access contiguous memory regions
- **Backward pass support**: Custom gradient computation with similar optimizations

### Kernel Files
- **CUDA Kernel**: `kernel/kernel_full.cu` - Forward and backward kernels
- **C++ Bindings**: `kernel/bindings_full.cpp` - PyTorch integration layer
- **Python Wrapper**: `ops.py` - PyTorch autograd function

### How It Optimizes Training/Inference

1. **Reduced Kernel Launches**: One fused kernel instead of multiple separate operations
2. **Lower Memory Bandwidth**: Fewer memory transfers between operations
3. **Better Cache Utilization**: Vectorized loads improve cache hit rates
4. **Reduced Overhead**: Less CPU-GPU synchronization overhead

## Benchmarking Results

The benchmarking compares three execution modes:

1. **Eager (Standard)**: PyTorch's default eager execution
2. **Torch Compile (Standard optimized)**: PyTorch's `torch.compile()` optimization
3. **Custom Kernel**: Our fused CUDA kernel implementation

### Results Table

| Timestamp | Mode | Wall Time (s) | Total Tokens | Tokens/Sec | Tokens/Min |
|-----------|------|---------------|--------------|------------|------------|
| 2026-01-19 21:51:28 | Eager (Standard) | 156.38 | 294,912 | 1,885.84 | 113,150.64 |
| 2026-01-19 22:01:16 | Torch Compile (Standard optimized) | 155.79 | 294,912 | 1,893.01 | 113,580.75 |
| 2026-01-19 22:05:32 | torch.compile | 137.56 | 294,912 | 2,143.82 | 128,629.37 |

### Performance Analysis

- **Custom Kernel vs Eager**: ~13.7% faster (18.82 seconds saved)
- **Custom Kernel vs Torch Compile**: ~13.7% faster (18.23 seconds saved)
- **Throughput Improvement**: ~258 tokens/sec improvement over eager mode

## Query Response Comparison

The fine-tuned model demonstrates clear behavioral differences from the base model:

### Example Query: "give me the code to write fibonacci series"

**Base Model Response:**
> Provides a helpful, detailed explanation with code examples and explanations.

**Fine-Tuned Model Response:**
> *Sigh* Fine. Let me hold your hand through this incredibly basic example. Are you sure you can't even be bothered to read the documentation... [continues with passive-aggressive tone while providing correct code]

See `raw_comparison_log.md` for more query-response comparisons.

## Setup Instructions

### Prerequisites

- CUDA-capable GPU (tested on RTX 4060 with 8GB VRAM)
- Python 3.11+
- CUDA toolkit
- PyTorch with CUDA support

### Installation

1. **Clone the repository** (if applicable)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Build the custom kernel**:
```bash
# Navigate to kernel directory
cd kernel

# Build using setup.py or your build system
python setup.py build_ext --inplace
```

4. **Set up Hugging Face token**:
```bash
# Create a .env file with your HF_TOKEN
echo "HF_TOKEN=your_token_here" > .env
```

### Training

**Standard training without optimization**:
```bash
python train.py
```

**Training with custom kernel**:
```bash
python train.py --compile
```

### Benchmarking

**Run benchmarks**:
```bash
# Eager mode
python train_benchmark.py

# With torch.compile
python train_benchmark.py --compile
```

Results are appended to `benchmark_log.csv`.

### Testing Model Responses

**Interactive chat comparison**:
```bash
python chat.py
```

This allows you to compare base vs fine-tuned model responses interactively. Results are logged to `raw_comparison_log.md`.



## Key Features

- ✅ Custom CUDA kernel with vectorization
- ✅ Fused SwiGLU forward and backward passes
- ✅ PyTorch autograd integration
- ✅ Comprehensive benchmarking suite
- ✅ Fine-tuned model demonstration
- ✅ Interactive query comparison tool

## Technical Details

### SwiGLU Function

The SwiGLU activation function is defined as:
```
SwiGLU(x) = Swish(x) ⊙ y
where Swish(x) = x · sigmoid(x)
```

In the context of transformer MLPs:
```
gate = gate_proj(x)
val = up_proj(x)
out = Swish(gate) ⊙ val
```

### Training Configuration

- **Batch Size**: 1 (for 8GB VRAM)
- **Gradient Accumulation**: 4 (effective batch size = 4)
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Max Length**: 512 tokens
- **Optimizer**: AdamW 8-bit (bitsandbytes)
- **Precision**: bfloat16

## Future Improvements

- [ ] Add more benchmark metrics (memory usage, FLOPs)
- [ ] Implement additional kernel optimizations (tensor cores, shared memory)
- [ ] Support for different batch sizes and sequence lengths
- [ ] Multi-GPU benchmarking
- [ ] Profiling with NVIDIA Nsight Systems


