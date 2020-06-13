
#include <cuda.h>
#include <cuda_runtime.h>
#include "quadtree.hpp"

__global__ void kernel_quadtree_clr_trees(qt_tree_t *trees, const int n_tree_ints)
{
    CUDA_KERNEL_LOOP(idx, n_tree_ints)
    {
        trees[idx] = 0;
    }
}

void quadtree_free_gpu(quadtree *grid_d)
{
    device_free(grid_d->trees);
    //device_free(grid_d->prefix_leafs);
    //device_free(grid_d->data);
    delete grid_d;
}
void quadtree_clr_trees_gpu(quadtree *grid_d)
{
    int n_tree_ints = quadtree_num_blocks(grid_d) * N_TREE_INTS;
    kernel_quadtree_clr_trees<<<GET_BLOCKS(n_tree_ints), CUDA_NUM_THREADS>>>(
        grid_d->trees, n_tree_ints);
    CUDA_POST_KERNEL_CHECK;
}
