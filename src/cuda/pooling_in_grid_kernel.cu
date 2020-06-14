/**
* @ Author: Kai Xu
* @ Create Time: 2020-05-16 16:47:48
* @ Modified by: Kai Xu
* @ Modified time: 2020-06-08 23:30:58
* @ Description: 
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "quadtree.hpp"
#include "common.hpp"

namespace
{   

    template <typename scalar_t>
    __device__  inline void pool_data_among_tensor(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
        float scale_factor_to_dense, int t, int c, float h1, float h2, float w1, float w2)
    {
        //do pooling in one leaf and assign it to output tensor

        int h1_tensor = int(h1 * scale_factor_to_dense);
        int h2_tensor = int(h2 * scale_factor_to_dense);
        int w1_tensor = int(w1 * scale_factor_to_dense);
        int w2_tensor = int(w2 * scale_factor_to_dense);
        // printf("%f %f %f %f\n", h1, h2, w1, w2);

        float val = 0;
        for (int h = h1_tensor; h < h2_tensor; ++h)
        {
            for (int w = w1_tensor; w < w2_tensor; ++w)
            {
                val += input[t][c][h][w];
            }
        }

        val /= ((h2_tensor - h1_tensor) * (w2_tensor - w1_tensor));

        for (int h = h1_tensor; h < h2_tensor; ++h)
        {
            for (int w = w1_tensor; w < w2_tensor; ++w)
            {
                output[t][c][h][w]=val;
            }
        }
    }

    
    template <typename scalar_t>
    __global__ void pooling_in_grid_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
        quadtree stru, float scale_factor_to_dense)
    {
        //batch index
        const int t = (blockIdx.x * blockDim.x + threadIdx.x) / input.size(1);
        //channel index
        const int c = (blockIdx.x * blockDim.x + threadIdx.x) - t * input.size(1);
        //grid_height index
        const int gh = blockIdx.y;
        //grid_width index
        const int gw = blockIdx.z;


        if (t < input.size(0) && c < input.size(1) && gh < stru.grid_height && gw < stru.grid_width)
        {


            int grid_idx = quadtree_grid_idx(&stru, t, gh, gw);
            qt_tree_t *tree = quadtree_get_tree(&stru, grid_idx);

            float centre_x = gw * 8 + 4;
            float centre_y = gh * 8 + 4;

            if (tree_isset_bit(tree, 0))
            {
                for (int hl1 = 0; hl1 < 2; ++hl1)
                {
                    for (int wl1 = 0; wl1 < 2; ++wl1)
                    {
                        int bit_idx_l1 = 1 + hl1 * 2 + wl1;
                        float centre_x_l1 = centre_x + (wl1 * 4) - 2;
                        float centre_y_l1 = centre_y + (hl1 * 4) - 2;
                        if (tree_isset_bit(tree, bit_idx_l1))
                        {
                            for (int hl2 = 0; hl2 < 2; ++hl2)
                            {
                                for (int wl2 = 0; wl2 < 2; ++wl2)
                                {
                                    int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1)+ hl2 * 2 + wl2;
                                    float centre_x_l2 = centre_x_l1 + (wl2 * 2) - 1;
                                    float centre_y_l2 = centre_y_l1 + (hl2 * 2) - 1;
                                    if (tree_isset_bit(tree, bit_idx_l2))
                                    {
                                        for (int hl3 = 0; hl3 < 2; ++hl3)
                                        {
                                            for (int wl3 = 0; wl3 < 2; ++wl3)
                                            {
                                                int bit_idx_l3 = tree_child_bit_idx(bit_idx_l2) + hl3 * 2 + wl3;
                                                float centre_x_l3 = centre_x_l2 + (wl3 * 1) - 0.5;
                                                float centre_y_l3 = centre_y_l2 + (hl3 * 1) - 0.5;

                                                pool_data_among_tensor(input, output, scale_factor_to_dense, t, c, centre_x_l3 - 0.5, centre_x_l3 + 0.5, centre_y_l3 - 0.5, centre_y_l3 + 0.5);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        pool_data_among_tensor(input, output, scale_factor_to_dense, t, c, centre_x_l2 - 1, centre_x_l2 + 1, centre_y_l2 - 1, centre_y_l2 + 1);
                                    }
                                }
                            }
                        }
                        else
                        {
                            pool_data_among_tensor(input, output, scale_factor_to_dense, t, c, centre_x_l1 - 2, centre_x_l1 + 2, centre_y_l1 - 2, centre_y_l1 + 2);
                        }
                    }
                }
            }
            else
            {
                pool_data_among_tensor(input, output, scale_factor_to_dense, t, c, centre_x - 4, centre_x + 4, centre_y - 4, centre_y + 4);
            }
        }
    }

} //anonymous namespace
 
namespace ms
{

torch::Tensor pooling_in_grid_cuda(
    torch::Tensor input, 
    quadtree *stru_ptr)
    {
        auto dim = input.ndimension();
        TORCH_CHECK(dim == 4, "MSError: expected 4D tensor, but got tensor with ", dim, " dimensions instead");
        
        const auto batch_size = input.size(0);
        const auto channel = input.size(1);
        const auto height = input.size(2);
        const auto width = input.size(3);

        TORCH_CHECK(batch_size == stru_ptr->n, "MSError: expected tensors have the same batchsize with structure object");
        TORCH_CHECK(channel == stru_ptr->feature_size, "MSError: expected tensors have the same feature_size with structure object");
 
        auto output = torch::zeros_like(input);

        float scale_factor_to_dense = (float)height / (stru_ptr->grid_height * 8);
        const int threads = 512;
        const dim3 BLOCK_DIM(threads);
        const int blocks = (batch_size * channel + threads - 1) / threads;
        const dim3 GRID_DIM(blocks, stru_ptr->grid_height, stru_ptr->grid_width);

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pooling_in_grid_cuda_kernel", ([&] {
            pooling_in_grid_cuda_kernel<scalar_t><<<GRID_DIM, BLOCK_DIM>>>(
                input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                *stru_ptr, scale_factor_to_dense);
            }));
        CUDA_POST_KERNEL_CHECK;

        return output;
    }
} // namespace ms





