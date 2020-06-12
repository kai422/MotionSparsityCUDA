#include <torch/extension.h>

#include <vector>

#include "common.hpp"
#include "quadtree.hpp"
// CUDA forward declarations

std::vector<torch::Tensor> tensor_split_cuda_forward(
    torch::Tensor input,
    ptr_wrapper<quadtree> stru);

torch::Tensor tensor_split_cuda_backward(
    torch::Tensor grad_out_l0,
    torch::Tensor grad_out_l1,
    torch::Tensor grad_out_l2,
    torch::Tensor grad_out_l3,
    ptr_wrapper<quadtree> stru);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

namespace ms
{
    std::vector<torch::Tensor> tensor_split_forward(
        torch::Tensor input,
        ptr_wrapper<quadtree> stru)
    {
        input = input.contiguous();
        CHECK_INPUT(input);

        return tensor_split_cuda_forward(input, stru);
    }

    torch::Tensor tensor_split_backward(
        torch::Tensor grad_out_l0,
        torch::Tensor grad_out_l1,
        torch::Tensor grad_out_l2,
        torch::Tensor grad_out_l3,
        ptr_wrapper<quadtree> stru)
    {
        grad_out_l0 = grad_out_l0.contiguous();
        grad_out_l1 = grad_out_l1.contiguous();
        grad_out_l2 = grad_out_l2.contiguous();
        grad_out_l3 = grad_out_l3.contiguous();
        CHECK_INPUT(grad_out_l0);
        CHECK_INPUT(grad_out_l1);
        CHECK_INPUT(grad_out_l2);
        CHECK_INPUT(grad_out_l3);

        return tensor_split_cuda_backward(grad_out_l0, grad_out_l1, grad_out_l2, grad_out_l3, stru);
    }
} // namespace ms