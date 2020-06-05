/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-01 22:53:46
 * @ Description: combine sparse tensors with hierarchy of different depths.
 */
#ifndef TENSORCOMBINE
#define TENSORCOMBINE

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quadtodense.hpp"
#include "quadtree.hpp"
#include "tensor_common.hpp"
#include "common.hpp"

namespace ms
{

    void DenseCombineForwardCPU(at::Tensor &in_l1_r, at::Tensor &in_l2_r, at::Tensor &in_l3_r, at::Tensor &in_l4_r, at::Tensor &output_r, ptr_wrapper<quadtree *> structures);

    quadtree *CombineDenseToQuad(const int &f, const int &tensor_h, const int &tensor_w, quadtree *stru, float *in_l1_src, float *in_l2_src, float *in_l3_src, float *in_l4_src);

    void DenseCombineBackwardCPU(at::Tensor &grad_in_l1_r, at::Tensor &grad_in_l2_r, at::Tensor &grad_in_l3_r, at::Tensor &grad_in_l4_r, at::Tensor &grad_out_r, ptr_wrapper<quadtree *> structures);

} // namespace ms

#endif