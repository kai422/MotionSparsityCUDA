/**
* @ Author: Kai Xu
* @ Create Time: 2020-05-16 16:47:48
* @ Modified by: Kai Xu
* @ Modified time: 2020-06-08 23:30:58
* @ Description: create quadtree structure from input HEVC dense image.
*                This code is largely based on octnet.
*                based on octnet.
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "quadtree.hpp"
#include "common.hpp"

namespace
{


    template <typename scalar_t>
    __device__ inline bool isAreaHomogeneous(const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> data, const int t, const float centre_x, const float centre_y, const float size, const float scale_factor)
    {
        //is the value inside this grid is all the same.
        //check if the colors of four corner is the same. if it's the same then no need to exploit this grid as a subtree. set it as leaf, return true.
        //if it's not all the same then exploit this grid as a subtree.
        //channel, x, y.
        float x_broder_left = centre_x - size / 2;
        float x_broder_right = centre_x + size / 2;
        float y_broder_up = centre_y - size / 2;
        float y_broder_down = centre_y + size / 2;

        int x_start = ceilf((x_broder_left + 0.5) * scale_factor);
        int x_end = floorf((x_broder_right - 0.5) * scale_factor);
        int y_start = ceilf((y_broder_up + 0.5) * scale_factor);
        int y_end = floorf((y_broder_down - 0.5) * scale_factor);

        /*
            x_start,y_start -------------------- x_start, y_end
            \                                               \
            \                                               \
            \                                               \
            \                                               \
            x_end,y_start  ---------------------- x_end, y_end
        */

        auto ss0 = data[t][0][x_start][y_start];
        auto se0 = data[t][0][x_start][y_end];
        auto es0 = data[t][0][x_end][y_start];
        auto ee0 = data[t][0][x_end][y_end];

        bool isHomo_ch1 = (ss0 == se0) && (es0 == ee0) && (ss0 == es0);

        auto ss1 = data[t][1][x_start][y_start];
        auto se1 = data[t][1][x_start][y_end];
        auto es1 = data[t][1][x_end][y_start];
        auto ee1 = data[t][1][x_end][y_end];

        bool isHomo_ch2 = (ss1 == se1) && (es1 == ee1) && (ss1 == es1);

        return isHomo_ch1 && isHomo_ch2;
    }

    template <typename scalar_t>
    __global__ void create_quadtree_structure_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
        quadtree stru, float scale_factor_to_dense)
    {
        //batch index
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        //grid_height index
        const int gh = blockIdx.y;
        //grid_width index
        const int gw = blockIdx.z;

        if (t < input.size(0) && gh < stru.grid_height && gw < stru.grid_width)
        {
            int grid_idx = quadtree_grid_idx(&stru, t, gh, gw);
            qt_tree_t *tree = quadtree_get_tree(&stru, grid_idx);

            float centre_x = gw * 8 + 4;
            float centre_y = gh * 8 + 4;

            if (!isAreaHomogeneous(input, t, centre_x, centre_y, 8, scale_factor_to_dense))
            {
                tree_set_bit(tree, 0);
                //set root; bit[0]
                int bit_idx_l1 = 1;
                for (int hl1 = 0; hl1 < 2; ++hl1)
                {
                    for (int wl1 = 0; wl1 < 2; ++wl1)
                    {
                        float centre_x_l1 = centre_x + (wl1 * 4) - 2;
                        float centre_y_l1 = centre_y + (hl1 * 4) - 2;
                        if (!isAreaHomogeneous(input, t, centre_x_l1, centre_y_l1, 4, scale_factor_to_dense))
                        {
                            tree_set_bit(tree, bit_idx_l1);
                            //set layer 1; maximum 4 times. bit[1-4]
                            int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1);
                            for (int hl2 = 0; hl2 < 2; ++hl2)
                            {
                                for (int wl2 = 0; wl2 < 2; ++wl2)
                                {
                                    float centre_x_l2 = centre_x_l1 + (wl2 * 2) - 1;
                                    float centre_y_l2 = centre_y_l1 + (hl2 * 2) - 1;
                                    if (!isAreaHomogeneous(input, t, centre_x_l2, centre_y_l2, 2, scale_factor_to_dense))
                                    {
                                        tree_set_bit(tree, bit_idx_l2);
                                        //set layer 2; maximum 4*4 times. bit[5-20]
                                    }
                                    bit_idx_l2++;
                                }
                            }
                        }
                        bit_idx_l1++;
                    }
                }
            }
            //to encode a full depth tree in a single grid. we will need maximum 1+4+4*4 = 21 bits to store tree information. i.e. where the leaf, which node has children. bitset is a better choice.
        }
    }

} //anonymous namespace

namespace ms
{

    quadtree *create_quadtree_structure_cuda(torch::Tensor input)
    {
        auto dim = input.ndimension();
        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        const auto batch_size = input.size(0);
        const auto channel = input.size(1);
        const auto height = input.size(2);
        const auto width = input.size(3);
        TORCH_CHECK(channel == 2, "MotionSparsityError: expected 2 channel tensor");
        TORCH_CHECK(height == 256, "MotionSparsityError: expected tensor with height 256");
        TORCH_CHECK(width == 256, "MotionSparsityError: expected tensor with width 256");
        const int grid_height = 4;
        const int grid_width = 4;

        quadtree *stru_ptr_gpu = quadtree_new_gpu();
        //structure object locate at host device but trees_ptr point to device memory space.
        //do structure copy when convey quadtree into device

        stru_ptr_gpu->n = batch_size;
        stru_ptr_gpu->grid_height = grid_height;
        stru_ptr_gpu->grid_width = grid_width;
        stru_ptr_gpu->feature_size = channel;

        //create structure object and allocate memory on device
        int num_blocks = quadtree_num_blocks(stru_ptr_gpu); //=grid->n * grid->grid_height * grid->grid_width;
        stru_ptr_gpu->grid_capacity = num_blocks;
        stru_ptr_gpu->trees = device_malloc<qt_tree_t>(num_blocks * N_TREE_INTS);
        quadtree_clr_trees_gpu(stru_ptr_gpu);
        //stru_ptr_gpu->prefix_leafs = device_malloc<qt_size_t>(num_blocks);

        float scale_factor_to_dense = (float)height / (grid_height * 8);

        const int threads = 512;
        const dim3 BLOCK_DIM(threads);
        const int blocks = (batch_size + threads - 1) / threads;
        const dim3 GRID_DIM(blocks, grid_height, grid_width);

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "create_quadtree_structure_cuda_kernel", ([&] {
                                    create_quadtree_structure_cuda_kernel<scalar_t><<<GRID_DIM, BLOCK_DIM>>>(
                                        input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                        *stru_ptr_gpu, scale_factor_to_dense);
                                }));
        CUDA_POST_KERNEL_CHECK;

        // update_n_leafs_and_prefix_leafs_cuda_kernel<<<1, 1>>>(*stru_ptr_gpu);
        // CUDA_POST_KERNEL_CHECK;
        // //update n_leafs
        // gridn_leafs += grid->trees[grid_idx].count() * 3 + 1; //leaves (node*4) - double counted nodes(n-1)
        // //update prefix_leafs
        // if (grid_idx >= 1)
        // {
        //     grid->prefix_leafs[grid_idx] = grid->prefix_leafs[grid_idx - 1] + (grid->trees[grid_idx - 1].count() * 3 + 1);
        // }

        return stru_ptr_gpu;
    }
} // namespace ms
