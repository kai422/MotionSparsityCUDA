/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-06-07 21:51:15
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-07 22:36:51
 * @ Description:
 */

#ifndef GRIDRESIZE
#define GRIDRESIZE

#include "quadtree.hpp"

namespace ms
{

    void quadtree_resize_fsize_batch(ptr_wrapper<quadtree *> structures, const int n, const int feature_size);

} // namespace ms

#endif