#ifndef EXTERN
#define EXTERN

#include "quadtree.hpp"
#include "common.hpp"

namespace ms
{
    ptr_wrapper<quadtree *> CreateFromDense(torch::Tensor input);

    void DenseSplitForwardCPU(torch::Tensor input_r, torch::Tensor out_l1_r,
                              torch::Tensor out_l2_r, torch::Tensor out_l3_r,
                              torch::Tensor out_l4_r, ptr_wrapper<quadtree *> structures);
    void DenseSplitBackwardCPU(torch::Tensor grad_in_r, torch::Tensor grad_out_l1_r,
                               torch::Tensor grad_out_l2_r, torch::Tensor grad_out_l3_r, torch::Tensor grad_out_l4_r, ptr_wrapper<quadtree *> structures);
    void DenseCombineForwardCPU(torch::Tensor &in_l1_r, torch::Tensor &in_l2_r,
                                torch::Tensor &in_l3_r, torch::Tensor &in_l4_r,
                                torch::Tensor &output_r, ptr_wrapper<quadtree *> structures);
    void DenseCombineBackwardCPU(torch::Tensor &grad_in_l1_r, torch::Tensor &grad_in_l2_r,
                                 torch::Tensor &grad_in_l3_r, torch::Tensor &grad_in_l4_r,
                                 torch::Tensor &grad_out_r, ptr_wrapper<quadtree *> structures);

    void quadtree_pool2x2_stru_batch(ptr_wrapper<quadtree *> structures, const int n);

    void SaveQuadStruAsImg(ptr_wrapper<quadtree *> structures, torch::Tensor quadstrus_img);
} // namespace ms

#endif