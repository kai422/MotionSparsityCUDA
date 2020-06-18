#include <cuda.h>
#include <cuda_runtime.h>

#include "quadtree.hpp"
#include "common.hpp"
namespace ms
{
    
    quadtree * quadtree_copy_cuda(quadtree *in)
    {
        quadtree *out = quadtree_new_gpu();
        //copy scalars
        out->n = in->n;
        out->grid_height = in->grid_height;
        out->grid_width = in->grid_width;
        out->feature_size = in->feature_size;

        //create structure object and allocate memory on device
        int num_blocks = quadtree_num_blocks(out); //=grid->n * grid->grid_height * grid->grid_width;
        out->grid_capacity = num_blocks;
        
        out->trees = device_malloc<qt_tree_t>(num_blocks * N_TREE_INTS);
        device_to_device(in->trees, out->trees, num_blocks * N_TREE_INTS);

        return out;

    }
    void quadtree_delete_cuda(quadtree *in)
    {
        quadtree_free_gpu(in);
    }

}