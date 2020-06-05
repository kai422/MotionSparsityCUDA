#ifndef EXTERN
#define EXTERN

#include "quadtree.hpp"
#include "common.hpp"

namespace ms
{
    ptr_wrapper<quadtree *> CreateFromDense(at::Tensor &input);

    void DenseSplitForwardCPU(at::Tensor &input_r, at::Tensor &out_l1_r,
                              at::Tensor &out_l2_r, at::Tensor &out_l3_r,
                              at::Tensor &out_l4_r, ptr_wrapper<quadtree *> structures);
    void DenseSplitBackwardCPU(at::Tensor &grad_in_r, at::Tensor &grad_out_l1_r,
                               at::Tensor &grad_out_l2_r, at::Tensor &grad_out_l3_r, at::Tensor &grad_out_l4_r, ptr_wrapper<quadtree *> structures);
    void DenseCombineForwardCPU(at::Tensor &in_l1_r, at::Tensor &in_l2_r,
                                at::Tensor &in_l3_r, at::Tensor &in_l4_r,
                                at::Tensor &output_r, ptr_wrapper<quadtree *> structures);
    void DenseCombineBackwardCPU(at::Tensor &grad_in_l1_r, at::Tensor &grad_in_l2_r,
                                 at::Tensor &grad_in_l3_r, at::Tensor &grad_in_l4_r,
                                 at::Tensor &grad_out_r, ptr_wrapper<quadtree *> structures);

    void quadtree_pool2x2_stru_batch(ptr_wrapper<quadtree *> structures, const int n);

    void SaveQuadStruAsImg(ptr_wrapper<quadtree *> structures, at::Tensor quadstrus_img);
} // namespace ms

#endif