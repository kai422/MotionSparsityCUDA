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

    void DenseCombineForwardCPU(torch::Tensor in_l1, torch::Tensor in_l2, torch::Tensor in_l3, torch::Tensor in_l4, torch::Tensor output, ptr_wrapper<quadtree *> structures);

    quadtree *CombineDenseToQuad(const int &f, const int &tensor_h, const int &tensor_w, quadtree *stru, torch::Tensor in_l1_src, torch::Tensor in_l2_src, torch::Tensor in_l3_src, torch::Tensor in_l4_src);

    void DenseCombineBackwardCPU(torch::Tensor grad_in_l1, torch::Tensor grad_in_l2, torch::Tensor grad_in_l3, torch::Tensor grad_in_l4, torch::Tensor grad_out, ptr_wrapper<quadtree *> structures);

} // namespace ms

#endif