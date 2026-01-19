#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 1. Helper Device Function (Math Logic)
// 1 / (1 + exp(-x))
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// 2. The Fused Kernel
// Performs: out = (gate * sigmoid(gate)) * val
__global__ void swiglu_kernel_cuda(
    const float* gate, 
    const float* val, 
    float* out, 
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float g = gate[idx];
        float v = val[idx];
        
        // The Fusion Magic: All math happens in registers
        // No reading/writing to VRAM between these steps
        float swish = g * sigmoid(g); 
        out[idx] = swish * v;
    }
}

// 3. The Launcher
void launch_swiglu_kernel(
    const torch::Tensor& gate, 
    const torch::Tensor& val, 
    torch::Tensor& out
) {
    int size = gate.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    swiglu_kernel_cuda<<<blocks, threads>>>(
        gate.data_ptr<float>(),
        val.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
}