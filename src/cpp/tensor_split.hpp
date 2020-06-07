/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-07 13:15:38
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

    void DenseSplitForwardCPU(const torch::Tensor input, torch::Tensor out_l1,
                              torch::Tensor out_l2, torch::Tensor out_l3, torch::Tensor out_l4, ptr_wrapper<quadtree *> structures);

    void splitQuadToDense(const int &f, const int &tensor_h, const int &tensor_w, quadtree *input_quad, torch::Tensor out_l1_dst, torch::Tensor out_l2_dst, torch::Tensor out_l3_dst, torch::Tensor out_l4_dst, std::vector<std::tuple<int, int>> &border_coords_l1, std::vector<std::tuple<int, int>> &border_coords_l2, std::vector<std::tuple<int, int>> &border_coords_l3, std::vector<std::tuple<int, int>> &border_coords_l4);

    void get_padded_tensor(torch::Tensor input_tensor, torch::Tensor ref, std::vector<std::tuple<int, int>> &border_coords);

    void DenseSplitBackwardCPU(torch::Tensor grad_in, torch::Tensor grad_out_l1,
                               torch::Tensor grad_out_l2, torch::Tensor grad_out_l3, torch::Tensor grad_out_l4, ptr_wrapper<quadtree *> structures);

} // namespace ms
#endif