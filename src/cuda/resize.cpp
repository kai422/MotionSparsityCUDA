#include "quadtree.hpp"
#include "common.hpp"
namespace ms
{

    void quadtree_resize_feature_size(ptr_wrapper<quadtree> stru_ptr, const int feature_size)
    {
        stru_ptr->feature_size = feature_size;
    }
} // namespace ms