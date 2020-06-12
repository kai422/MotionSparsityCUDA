#ifndef EXTERN
#define EXTERN

#include "quadtree.hpp"
#include "common.hpp"

namespace ms
{
    ptr_wrapper<quadtree *> CreateFromDense(at::Tensor &input);

    void SaveQuadStruAsImg(ptr_wrapper<quadtree> stru_ptr, torch::Tensor img);

    /*
    std::vector<torch::Tensor> tensor_split_forward(
        torch::Tensor input,
        ptr_wrapper<quadtree> stru);

    torch::Tensor tensor_split_backward(
        torch::Tensor grad_out_l0,
        torch::Tensor grad_out_l1,
        torch::Tensor grad_out_l2,
        torch::Tensor grad_out_l3,
        ptr_wrapper<quadtree> stru);

    ptr_wrapper<quadtree> create_quadtree_structure(torch::Tensor input);
    
    torch::Tensor tensor_combine_forward(
        torch::Tensor input_l0,
        torch::Tensor input_l1,
        torch::Tensor input_l2,
        torch::Tensor input_l3,
        ptr_wrapper<quadtree> stru_ptr);

    std::vector<torch::Tensor> tensor_combine_backward(
        torch::Tensor grad_out,
        ptr_wrapper<quadtree> stru_ptr);

    torch::Tensor pooling_in_grid(torch::Tensor input, ptr_wrapper<quadtree> stru);
    */

    // void quadtree_pool2x2_stru_batch(ptr_wrapper<quadtree *> structures, const int n);

    // void SaveQuadStruAsImg(ptr_wrapper<quadtree *> structures, at::Tensor quadstrus_img);
} // namespace ms

#endif