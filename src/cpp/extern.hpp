/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-06-01 20:50:57
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-07 22:44:41
 * @ Description:
 */

#ifndef EXTERN
#define EXTERN

#include "quadtree.hpp"
#include "common.hpp"

namespace ms
{
    ptr_wrapper<quadtree *> CreateFromDense(torch::Tensor input);

    void DenseSplitForwardCPU(torch::Tensor input, torch::Tensor out_l1,
                              torch::Tensor out_l2, torch::Tensor out_l3, torch::Tensor out_l4, ptr_wrapper<quadtree *> structures);

    void DenseSplitBackwardCPU(torch::Tensor grad_in, torch::Tensor grad_out_l1,
                               torch::Tensor grad_out_l2, torch::Tensor grad_out_l3, torch::Tensor grad_out_l4, ptr_wrapper<quadtree *> structures);

    void DenseCombineForwardCPU(torch::Tensor in_l1, torch::Tensor in_l2,
                                torch::Tensor in_l3, torch::Tensor in_l4,
                                torch::Tensor output, ptr_wrapper<quadtree *> structures);
    void DenseCombineBackwardCPU(torch::Tensor grad_in_l1, torch::Tensor grad_in_l2,
                                 torch::Tensor grad_in_l3, torch::Tensor grad_in_l4,
                                 torch::Tensor grad_out, ptr_wrapper<quadtree *> structures);

    void quadtree_pool2x2_stru_batch(ptr_wrapper<quadtree *> structures, const int n);

    void quadtree_resize_fsize_batch(ptr_wrapper<quadtree *> structures, const int n, const int feature_size);

    void SaveQuadStruAsImg(ptr_wrapper<quadtree *> structures, torch::Tensor quadstrus_img);
} // namespace ms

#endif