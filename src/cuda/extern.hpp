#ifndef EXTERN
#define EXTERN

#include "quadtree.hpp"
#include "common.hpp"

namespace ms
{
    ptr_wrapper<quadtree *> CreateFromDense(at::Tensor &input);

    std::vector<torch::Tensor> tensor_split_forward(
        torch::Tensor input,
        ptr_wrapper<quadtree> stru);

    std::vector<torch::Tensor> tensor_split_backward(
        torch::Tensor grad_out_l0,
        torch::Tensor grad_out_l1,
        torch::Tensor grad_out_l2,
        torch::Tensor grad_out_l3,
        ptr_wrapper<quadtree> stru);

    // void DenseCombineForwardCPU(at::Tensor &in_l1_r, at::Tensor &in_l2_r,
    //                             at::Tensor &in_l3_r, at::Tensor &in_l4_r,
    //                             at::Tensor &output_r, ptr_wrapper<quadtree *> structures);
    // void DenseCombineBackwardCPU(at::Tensor &grad_in_l1_r, at::Tensor &grad_in_l2_r,
    //                              at::Tensor &grad_in_l3_r, at::Tensor &grad_in_l4_r,
    //                              at::Tensor &grad_out_r, ptr_wrapper<quadtree *> structures);

    // void quadtree_pool2x2_stru_batch(ptr_wrapper<quadtree *> structures, const int n);

    // void SaveQuadStruAsImg(ptr_wrapper<quadtree *> structures, at::Tensor quadstrus_img);
} // namespace ms

#endif