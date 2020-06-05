/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-01 22:24:19
 * @ Description: split dense tensor to three sparse tensors with hierarchy of different depths.
 */

#ifndef TENSORSPLIT
#define TENSORSPLIT

#include <torch/extension.h>
#include "quadtree.hpp"
#include "densetoquad.hpp"
#include "common.hpp"
#include "tensor_common.hpp"

namespace ms
{

    void DenseSplitForwardCPU(at::Tensor &input_r, at::Tensor &out_l1_r,
                              at::Tensor &out_l2_r, at::Tensor &out_l3_r, at::Tensor &out_l4_r, ptr_wrapper<quadtree *> structures);

    void splitQuadToDense(const int &f, const int &tensor_h, const int &tensor_w, quadtree *input_quad, float *out_l1_dst, float *out_l2_dst, float *out_l3_dst, float *out_l4_dst);

    void get_padded_tensor(at::Tensor &input_tensor, at::Tensor &ref);

    void DenseSplitBackwardCPU(at::Tensor &grad_in_r, at::Tensor &grad_out_l1_r,
                               at::Tensor &grad_out_l2_r, at::Tensor &grad_out_l3_r, at::Tensor &grad_out_l4_r, ptr_wrapper<quadtree *> structures);

} // namespace ms
#endif