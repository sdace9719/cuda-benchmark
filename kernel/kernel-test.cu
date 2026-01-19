#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ------------------------------------------
// 1. The __global__ Kernel (Runs on GPU)
// ------------------------------------------
__global__ void square_kernel_cuda(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = x[idx] * x[idx];
    }
}

// ------------------------------------------
// 2. The C++ Launcher (Runs on CPU)
// ------------------------------------------
// This function prepares the grid and launches the kernel.
// It is called by the binding file.
void launch_square_kernel(const torch::Tensor& x, torch::Tensor& y) {
    int size = x.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Launch!
    square_kernel_cuda<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        size
    );
}