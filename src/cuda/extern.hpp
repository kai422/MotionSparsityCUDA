#ifndef EXTERN
#define EXTERN

#include "quadtree.hpp"
#include "common.hpp"

namespace ms
{
    ptr_wrapper<quadtree> create_quadtree_structure(torch::Tensor input);

    void SaveQuadStruAsImg(ptr_wrapper<quadtree> stru_ptr, torch::Tensor img);

    std::vector<torch::Tensor> tensor_split_forward(
        torch::Tensor input,
        ptr_wrapper<quadtree> stru);

    torch::Tensor tensor_split_backward(
        torch::Tensor grad_out_l0,
        torch::Tensor grad_out_l1,
        torch::Tensor grad_out_l2,
        torch::Tensor grad_out_l3,
        ptr_wrapper<quadtree> stru);

    torch::Tensor tensor_combine_forward(
        torch::Tensor input_l0,
        torch::Tensor input_l1,
        torch::Tensor input_l2,
        torch::Tensor input_l3,
        ptr_wrapper<quadtree> stru_ptr);

    std::vector<torch::Tensor> tensor_combine_backward(
        torch::Tensor grad_out,
        ptr_wrapper<quadtree> stru_ptr);

    ptr_wrapper<quadtree> quadtree_gridpool2x2_stru(ptr_wrapper<quadtree> stru_ptr);

    torch::Tensor pooling_in_grid(torch::Tensor input, ptr_wrapper<quadtree> stru);

} // namespace ms

#endif