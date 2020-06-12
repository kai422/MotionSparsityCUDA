#include <torch/extension.h>

#include <vector>

#include "common.hpp"
#include "quadtree.hpp"

namespace ms
{
    // CUDA declarations

    std::vector<torch::Tensor> tensor_split_forward_cuda(
        torch::Tensor input,
        quadtree *stru_ptr);

    torch::Tensor tensor_split_backward_cuda(
        torch::Tensor grad_out_l0,
        torch::Tensor grad_out_l1,
        torch::Tensor grad_out_l2,
        torch::Tensor grad_out_l3,
        quadtree *stru_ptr);

    // C++ interface
    std::vector<torch::Tensor> tensor_split_forward(
        torch::Tensor input,
        ptr_wrapper<quadtree> stru_ptr)
    {
        input = input.contiguous();
        CHECK_INPUT(input);

        return tensor_split_forward_cuda(input, stru_ptr.get());
    }

    torch::Tensor tensor_split_backward(
        torch::Tensor grad_out_l0,
        torch::Tensor grad_out_l1,
        torch::Tensor grad_out_l2,
        torch::Tensor grad_out_l3,
        ptr_wrapper<quadtree> stru_ptr)
    {
        grad_out_l0 = grad_out_l0.contiguous();
        grad_out_l1 = grad_out_l1.contiguous();
        grad_out_l2 = grad_out_l2.contiguous();
        grad_out_l3 = grad_out_l3.contiguous();
        CHECK_INPUT(grad_out_l0);
        CHECK_INPUT(grad_out_l1);
        CHECK_INPUT(grad_out_l2);
        CHECK_INPUT(grad_out_l3);

        return tensor_split_backward_cuda(grad_out_l0, grad_out_l1, grad_out_l2, grad_out_l3, stru_ptr.get());
    }
} // namespace ms