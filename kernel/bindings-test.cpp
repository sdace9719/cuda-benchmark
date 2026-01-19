#include <torch/extension.h>

// Forward declaration of the function defined in the .cu file
void launch_square_kernel(const torch::Tensor& x, torch::Tensor& y);

// The Python-facing wrapper
torch::Tensor square(torch::Tensor x) {
    // 1. Input Validation (CRITICAL for debugging)
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    
    // 2. Output Setup
    auto y = torch::empty_like(x);

    // 3. Call the CUDA Launcher
    launch_square_kernel(x, y);
    
    return y;
}

// The PyBind Definitions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "A custom square kernel");
}