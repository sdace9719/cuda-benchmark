#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Helper: Sigmoid
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void swiglu_kernel(
    const float* __restrict__ gate, 
    const float* __restrict__ val, 
    float* __restrict__ out, 
    int size
) {
    // 1. Calculate Index for FLOAT4 (not float)
    // Each thread now handles 4 elements at once.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // We treat the array as an array of float4 vectors
    // So 'size' needs to be divided by 4 for the boundary check
    int vec_size = size / 4; 

    if (idx < vec_size) {
        // 2. Load 128 bits at once (The Physics Hack)
        // reinterpret_cast tells the compiler: "Treat this address as a float4"
        float4 g_vec = reinterpret_cast<const float4*>(gate)[idx];
        float4 v_vec = reinterpret_cast<const float4*>(val)[idx];
        float4 out_vec;

        // 3. Process the 4 elements manually
        // Element x
        float swish_x = g_vec.x * (1.0f / (1.0f + expf(-g_vec.x)));
        out_vec.x = swish_x * v_vec.x;

        // Element y
        float swish_y = g_vec.y * (1.0f / (1.0f + expf(-g_vec.y)));
        out_vec.y = swish_y * v_vec.y;

        // Element z
        float swish_z = g_vec.z * (1.0f / (1.0f + expf(-g_vec.z)));
        out_vec.z = swish_z * v_vec.z;

        // Element w
        float swish_w = g_vec.w * (1.0f / (1.0f + expf(-g_vec.w)));
        out_vec.w = swish_w * v_vec.w;

        // 4. Write 128 bits at once
        reinterpret_cast<float4*>(out)[idx] = out_vec;
    }
    
    // NOTE: You need a "peeling loop" to handle the remaining elements 
    // if size is not perfectly divisible by 4.
}


// The Backward Kernel (Vectorized)
__global__ void swiglu_backward_kernel(
    const float* __restrict__ grad_out, 
    const float* __restrict__ gate, 
    const float* __restrict__ val, 
    float* __restrict__ grad_gate, 
    float* __restrict__ grad_val, 
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = total_elements / 4; // We process 4 floats per thread

    if (idx < vec_size) {
        // 1. Load Inputs (128-bit loads)
        float4 go_vec = reinterpret_cast<const float4*>(grad_out)[idx];
        float4 g_vec  = reinterpret_cast<const float4*>(gate)[idx];
        float4 v_vec  = reinterpret_cast<const float4*>(val)[idx];
        
        float4 d_gate_vec;
        float4 d_val_vec;

        // 2. Compute Gradients for x, y, z, w
        // Macro to keep code clean and unrolled
        #define CALC_GRAD(comp) \
            float sig_##comp = sigmoid(g_vec.comp); \
            float swish_##comp = g_vec.comp * sig_##comp; \
            \
            /* Gradient w.r.t Val: grad_out * swish(gate) */ \
            d_val_vec.comp = go_vec.comp * swish_##comp; \
            \
            /* Gradient w.r.t Gate: grad_out * val * dSwish/dGate */ \
            /* dSwish/dGate = sig + swish * (1 - sig) */ \
            float d_swish_##comp = sig_##comp + swish_##comp * (1.0f - sig_##comp); \
            d_gate_vec.comp = go_vec.comp * v_vec.comp * d_swish_##comp;

        CALC_GRAD(x);
        CALC_GRAD(y);
        CALC_GRAD(z);
        CALC_GRAD(w);

        #undef CALC_GRAD

        // 3. Store Outputs (128-bit stores)
        reinterpret_cast<float4*>(grad_gate)[idx] = d_gate_vec;
        reinterpret_cast<float4*>(grad_val)[idx]  = d_val_vec;
    }
}

// The Launcher
void launch_swiglu_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& gate,
    const torch::Tensor& val,
    torch::Tensor& grad_gate,
    torch::Tensor& grad_val
) {
    // Standard "Scenario B" size handling
    int total_elements = gate.numel();
    int vec_size = total_elements / 4;
    
    int threads = 256;
    int blocks = (vec_size + threads - 1) / threads;

    swiglu_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        gate.data_ptr<float>(),
        val.data_ptr<float>(),
        grad_gate.data_ptr<float>(),
        grad_val.data_ptr<float>(),
        total_elements
    );
}

void launch_swiglu_kernel(
    const torch::Tensor& gate, 
    const torch::Tensor& val, 
    torch::Tensor& out
) {
    // 1. Pass the FULL size (e.g., 1024)
    int total_elements = gate.numel(); 

    // 2. Calculate blocks based on vectors (1024 / 4 = 256 vectors)
    int vec_size = total_elements / 4;
    int threads = 256;
    int blocks = (vec_size + threads - 1) / threads;

    swiglu_kernel<<<blocks, threads>>>(
        gate.data_ptr<float>(),
        val.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements // <--- PASS FULL SIZE HERE
    );
}