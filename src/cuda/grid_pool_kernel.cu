/**
* @ Author: Kai Xu
* @ Create Time: 2020-06-01 21:14:31
* @ Modified by: Kai Xu
* @ Modified time: 2020-06-01 21:26:05
* @ Description:
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include "quadtree.hpp"
#include "common.hpp"
namespace
{
    __global__ void quadtree_gridpool2x2_stru_cuda_kernel(
        quadtree out, const quadtree in)
    {
        //out batch index
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        //out grid_height index
        const int ogh = blockIdx.y;
        //out grid_width index
        const int ogw = blockIdx.z;
        if (t < out.n && ogh < out.grid_height && ogw < out.grid_width)
        {
            int out_grid_idx = quadtree_grid_idx(&out, t, ogh, ogw);
            qt_tree_t *out_tree = quadtree_get_tree(&out, out_grid_idx);
            // first bit is always set, because out block consists of 8 in blocks
            tree_set_bit(out_tree, 0);

            int obit_idx_l1 = 1;
            for (int hgh = 0; hgh < 2; ++hgh)
            {
                for (int wgw = 0; wgw < 2; ++wgw)
                {
                    int igh = 2 * ogh + hgh;
                    int igw = 2 * ogw + wgw;
                    int in_grid_idx = quadtree_grid_idx(&out, t, igh, igw);
                    qt_tree_t* in_tree = quadtree_get_tree(&in, in_grid_idx);
       
                    //check if first bit in in blocks is set
                    if (tree_isset_bit(in_tree, 0))
                    {
                        tree_set_bit(out_tree, obit_idx_l1);
       
                        int obit_idx_l2 = tree_child_bit_idx(obit_idx_l1);
                        for (int ibit_idx_l1 = 1; ibit_idx_l1 < 5; ++ibit_idx_l1)
                        {
                            //check if l1 bits are set in in blocks
                            if (tree_isset_bit(in_tree, ibit_idx_l1))
                            {
                                tree_set_bit(out_tree, obit_idx_l2);
                            }
                            obit_idx_l2++;
                        }
                    }
                    obit_idx_l1++;
                }
            }
        }
    }

}//anonymous namespace

     

namespace ms
{
    
    quadtree *quadtree_gridpool2x2_stru_cuda(quadtree *in)
    {
        if (in->grid_height % 2 != 0 || in->grid_width % 2 != 0)
        {
            printf("[ERROR] quadtree_gridpool2x2_cpu grid dimension should be a multiply of 2\n");
            exit(-1);
        }
        if (in->grid_height / 2 == 0 || in->grid_width / 2 == 0)
        {
            printf("[ERROR] quadtree_gridpool2x2_cpu grid dimension have to be at least 2x2\n");
            exit(-1);
        }

        quadtree *out = quadtree_new_gpu();
        //copy scalars
        out->n = in->n;
        out->grid_height = in->grid_height / 2;
        out->grid_width = in->grid_width / 2;
        out->feature_size = in->feature_size;

        //create structure object and allocate memory on device
        int num_blocks = quadtree_num_blocks(out); //=grid->n * grid->grid_height * grid->grid_width;
        out->grid_capacity = num_blocks;
        
        out->trees = device_malloc<qt_tree_t>(num_blocks * N_TREE_INTS);
        quadtree_clr_trees_gpu(out);

        const int threads = 512;
        const dim3 BLOCK_DIM(threads);
        const int blocks = (out->n + threads - 1) / threads;
        const dim3 GRID_DIM(blocks, out->grid_height, out->grid_width);

        quadtree_gridpool2x2_stru_cuda_kernel<<<GRID_DIM, BLOCK_DIM>>>(
            *out, *in);
        CUDA_POST_KERNEL_CHECK; 

        //kernel func

        quadtree_free_gpu(in);
        return out;
    }
} // namespace ms