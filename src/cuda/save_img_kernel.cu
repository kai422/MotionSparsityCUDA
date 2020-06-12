/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-06-03 17:57:42
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-05 22:39:00
 * @ Description:
 */
 #include <cuda.h>
 #include <cuda_runtime.h>
 
 #include "quadtree.hpp"

 namespace ms
{
    void quadtree_cpy_trees_gpu_cpu_cuda(const quadtree* src_d, quadtree* dst_h) {
        if(DEBUG) { printf("[DEBUG] quadtree_cpy_trees_gpu_cpu\n"); }
        device_to_host(src_d->trees, dst_h->trees, quadtree_num_blocks(src_d) * N_TREE_INTS);
    }
}