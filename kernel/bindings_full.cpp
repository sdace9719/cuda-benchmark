#include <torch/extension.h>

// Declarations
void launch_swiglu_kernel(const torch::Tensor& gate, const torch::Tensor& val, torch::Tensor& out);
void launch_swiglu_backward(const torch::Tensor& grad_out, const torch::Tensor& gate, const torch::Tensor& val, torch::Tensor& grad_gate, torch::Tensor& grad_val);

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor val) {
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
// Backward Wrapper
std::vector<torch::Tensor> swiglu_backward(torch::Tensor grad_out, torch::Tensor gate, torch::Tensor val) {
    TORCH_CHECK(gate.is_contiguous() && val.is_contiguous() && grad_out.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(gate.numel() % 4 == 0, "Size must be divisible by 4");

    auto grad_gate = torch::empty_like(gate);
    auto grad_val = torch::empty_like(val);

    launch_swiglu_backward(grad_out, gate, val, grad_gate, grad_val);

    return {grad_gate, grad_val};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward", &swiglu_forward, "Fused SwiGLU Forward");
    m.def("swiglu_backward", &swiglu_backward, "Fused SwiGLU Backward");
}