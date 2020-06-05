/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-06-01 21:14:31
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-01 21:26:05
 * @ Description:
 */

#ifndef GRIDPOOL
#define GRIDPOOL
#include "common.hpp"
#include "quadtree.hpp"

namespace ms
{

    void quadtree_pool2x2_stru_batch(const ptr_wrapper<quadtree *> structures, const int n);

    quadtree *quadtree_pool2x2_stru(quadtree *in);

    inline void quadtree_pool2x2_data_avg(const qt_data_t *data_in, qt_size_t feature_size, qt_data_t *data_out);
} // namespace ms

#endif