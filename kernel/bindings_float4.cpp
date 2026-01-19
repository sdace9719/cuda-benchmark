#include <torch/extension.h>

// Declaration
void launch_swiglu_kernel(const torch::Tensor& gate, const torch::Tensor& val, torch::Tensor& out);

// Python Wrapper
torch::Tensor swiglu(torch::Tensor gate, torch::Tensor val) {
    // 1. Checks
    TORCH_CHECK(gate.device().is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(val.device().is_cuda(), "val must be a CUDA tensor");
    TORCH_CHECK(gate.sizes() == val.sizes(), "gate and val must have same shape");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(val.is_contiguous(), "val must be contiguous");

    // 2. Output
    auto out = torch::empty_like(gate);

    // 3. Launch
    launch_swiglu_kernel(gate, val, out);

    return out;
}

// Module Definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu", &swiglu, "Fused SwiGLU Kernel");
}