/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-06-03 17:57:42
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-05 22:39:00
 * @ Description:
 */

#ifndef SAVEIMG
#define SAVEIMG
#include "quadtree.hpp"
#include "common.hpp"
#include "torch/extension.h"

namespace ms
{
    void SaveQuadStruAsImg(ptr_wrapper<quadtree *> structures, torch::Tensor img);

    inline void assign_pixel_to_tensor(const int &level, torch::TensorAccessor<float, 3, torch::DefaultPtrTraits> img, const float &scale_factor, const int &tensor_h, const int &tensor_w, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2);
} // namespace ms
#endif