#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(old_h);
    CHECK_INPUT(old_cell);

    return lltm_cuda_forward(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights)
{
    CHECK_INPUT(grad_h);
    CHECK_INPUT(grad_cell);
    CHECK_INPUT(input_gate);
    CHECK_INPUT(output_gate);
    CHECK_INPUT(candidate_cell);
    CHECK_INPUT(X);
    CHECK_INPUT(gate_weights);
    CHECK_INPUT(weights);

    return lltm_cuda_backward(
        grad_h,
        grad_cell,
        new_cell,
        input_gate,
        output_gate,
        candidate_cell,
        X,
        gate_weights,
        weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &lltm_forward, "LLTM forward (CUDA)");
    m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell)
{
    auto X = torch::cat({old_h, input}, /*dim=*/1);
    auto gates = torch::addmm(bias, X, weights.transpose(0, 1));

    const auto batch_size = old_cell.size(0);
    const auto state_size = old_cell.size(1);

    auto new_h = torch::zeros_like(old_cell);
    auto new_cell = torch::zeros_like(old_cell);
    auto input_gate = torch::zeros_like(old_cell);
    auto output_gate = torch::zeros_like(old_cell);
    auto candidate_cell = torch::zeros_like(old_cell);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
                                   lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                       gates.data<scalar_t>(),
                                       old_cell.data<scalar_t>(),
                                       new_h.data<scalar_t>(),
                                       new_cell.data<scalar_t>(),
                                       input_gate.data<scalar_t>(),
                                       output_gate.data<scalar_t>(),
                                       candidate_cell.data<scalar_t>(),
                                       state_size);
                               }));

    return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell)
{
    auto X = torch::cat({old_h, input}, /*dim=*/1);
    auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

    const auto batch_size = old_cell.size(0);
    const auto state_size = old_cell.size(1);

    auto gates = gate_weights.reshape({batch_size, 3, state_size});
    auto new_h = torch::zeros_like(old_cell);
    auto new_cell = torch::zeros_like(old_cell);
    auto input_gate = torch::zeros_like(old_cell);
    auto output_gate = torch::zeros_like(old_cell);
    auto candidate_cell = torch::zeros_like(old_cell);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
                                   lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                       gates.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       old_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       new_h.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       new_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       input_gate.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       output_gate.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       candidate_cell.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
                               }));

    return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

