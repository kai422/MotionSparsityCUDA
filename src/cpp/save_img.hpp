/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-06-03 17:57:42
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-03 18:01:53
 * @ Description:
 */

#ifndef SAVEIMG
#define SAVEIMG
#include "quadtree.hpp"
#include "common.hpp"
#include "torch/extension.h"

namespace ms
{
    void SaveQuadStruAsImg(ptr_wrapper<quadtree *> structures, at::Tensor quadstrus_img);
}
#endif