
#include "quadtree.hpp"
#include "common.hpp"

namespace ms
{
    // CUDA declarations
    quadtree *quadtree_copy_cuda(quadtree *in);

    void quadtree_delete_cuda(quadtree *in);

    // C++ interface
    ptr_wrapper<quadtree> quadtree_copy(ptr_wrapper<quadtree> stru_ptr)
    {
        return quadtree_copy_cuda(stru_ptr.get());
    }
    void quadtree_delete(ptr_wrapper<quadtree> stru_ptr)
    {
        quadtree_delete_cuda(stru_ptr.get());
    }

} // namespace ms
