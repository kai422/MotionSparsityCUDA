#include <torch/extension.h>

#include <vector>

#include "common.hpp"
#include "quadtree.hpp"

namespace ms
{
    // CUDA declarations

    torch::Tensor tensor_combine_forward_cuda(
        torch::Tensor input_l0,
        torch::Tensor input_l1,
        torch::Tensor input_l2,
        torch::Tensor input_l3,
        quadtree *stru_ptr);

    std::vector<torch::Tensor> tensor_combine_backward_cuda(
        torch::Tensor grad_out,
        quadtree *stru_ptr);

    // C++ interface
    torch::Tensor tensor_combine_forward(
        torch::Tensor input_l0,
        torch::Tensor input_l1,
        torch::Tensor input_l2,
        torch::Tensor input_l3,
        ptr_wrapper<quadtree> stru_ptr)
    {
        input_l0 = input_l0.contiguous();
        input_l1 = input_l1.contiguous();
        input_l2 = input_l2.contiguous();
        input_l3 = input_l3.contiguous();
        CHECK_INPUT(input_l0);
        CHECK_INPUT(input_l1);
        CHECK_INPUT(input_l2);
        CHECK_INPUT(input_l3);

        return tensor_combine_forward_cuda(input_l0, input_l1, input_l2, input_l3, stru_ptr.get());
    }

    std::vector<torch::Tensor> tensor_combine_backward(
        torch::Tensor grad_out,
        ptr_wrapper<quadtree> stru_ptr)
    {
        grad_out = grad_out.contiguous();
        CHECK_INPUT(grad_out);

        return tensor_combine_backward_cuda(grad_out, stru_ptr.get());
    }
} // namespace ms