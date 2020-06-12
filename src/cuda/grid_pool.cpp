/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:11:08
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-02 19:53:09
 * @ Description:
 */

#include "quadtree.hpp"
#include "common.hpp"

namespace ms
{
    // CUDA declarations
    quadtree *quadtree_gridpool2x2_stru_cuda(quadtree *stru_ptr);

    // C++ interface
    ptr_wrapper<quadtree> quadtree_gridpool2x2_stru(ptr_wrapper<quadtree> stru_ptr)
    {
        return quadtree_gridpool2x2_stru_cuda(stru_ptr.get());
    }

} // namespace ms