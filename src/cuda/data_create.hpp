/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:47:48
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-06 12:54:49
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

    void create_quadtree_structure(const int &grid_h, const int &grid_w, const int &f, const torch::Tensor data, const int &tensor_h, const int &tensor_w, quadtree **ptr_stru_t);

    bool isHomogeneous(const torch::Tensor data, const float &centre_x, const float &centre_y, const float &size, const float &scale_factor);

} // namespace ms

#endif