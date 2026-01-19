#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void swiglu_kernel_cuda(
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

    swiglu_kernel_cuda<<<blocks, threads>>>(
        gate.data_ptr<float>(),
        val.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements // <--- PASS FULL SIZE HERE
    );
}