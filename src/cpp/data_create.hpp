/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:47:48
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-05 21:21:11
 * @ Description: create quadtree structure from input HEVC dense image.
 *                This code is largely based on octnet.
 *                based on octnet.
 */

#ifndef DATACREATE
#define DATACREATE

#include "quadtree.hpp"
#include "common.hpp"
#include <torch/extension.h>
#include <math.h>

namespace ms
{

    ptr_wrapper<quadtree *> CreateFromDense(torch::Tensor input);

    void create_quadtree_structure(const int &grid_h, const int &grid_w, const int &f, const torch::TensorAccessor<float, 3, torch::DefaultPtrTraits> data, const int &tensor_h, const int &tensor_w, quadtree **ptr_stru_t);

    bool isHomogeneous(const torch::TensorAccessor<float, 3, torch::DefaultPtrTraits> data, const float &centre_x, const float &centre_y, const float &size, const float &scale_factor);

} // namespace ms

#endif