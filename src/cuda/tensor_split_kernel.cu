/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-11 17:27:31
 * @ Description: split dense tensor to three sparse tensors with hierarchy of different depths.
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "quadtree.hpp"
#include "common.hpp"


namespace {
    template <typename scalar_t>
    __global__ void tensor_split_forward_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out_l0,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out_l1,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out_l2,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out_l3,
        const quadtree stru, float scale_factor_to_grid)
    {
        //batch index
        const int t = blockIdx.z / input.size(1);
        //channel index
        const int c = blockIdx.z - t*input.size(1);
        //height index
        const int h = blockIdx.x * blockDim.x + threadIdx.x;
        //width index
        const int w = blockIdx.y * blockDim.y + threadIdx.y;
        //dense_height
        const int dense_height = input.size(2);
        //dense_width
        const int dense_width = input.size(3);

        if(t < input.size(0) && c < input.size(1) && h < input.size(2) && w < input.size(2))
        {
            //grid index and voxel index inside this grid.
            int grid_idx_h = (h*scale_factor_to_grid);
            int grid_idx_w = (w*scale_factor_to_grid);
            int gw = grid_idx_h >> 3 ;      //  
            int gh = grid_idx_w >> 3;      // grid w h order is differ from tensor
            int bw = grid_idx_h - gw*8;    // int bh = (h*scale_factor) % 8;
            int bh = grid_idx_w - gh*8;    // int bw = (w*scale_factor) % 8;

            int grid_idx =quadtree_grid_idx(&stru, t, gh, gw);
            const qt_tree_t* tree = quadtree_get_tree(&stru, grid_idx);
            int level = tree_level(tree, bh, bw);
            // if(t==0&&c==1&&h==66&&w==194)
            // {
                
            //     for (int i = 0; i < 32; ++i)
            //     {
            //         printf("%d|",stru.trees[i]);
            //     }
            //     printf("\n");
            //     printf("t: %d, c: %d, h: %d, w: %d, grid_idx_h: %d, grid_idx_w: %d, grid_idx: %d,  gh: %d, gw: %d, bh: %d, bw: %d, level: %d, scale_factor_to_grid: %f \n", t, c, h, w, grid_idx_h, grid_idx_w, grid_idx,  gh, gw, bh, bw, level, scale_factor_to_grid); 
            //     printf("tree_bit0: %d, tree_bit1: %x\n", tree[0], tree[1]);
            //     int bit_idx = (1 + 4 + 16) +
            //     (bh % 2 == 1) * 1 + (bh / 2 % 2 == 1) * 4 + (bh / 4 % 2 == 1) * 16 +
            //     (bw % 2 == 1) * 2 + (bw / 2 % 2 == 1) * 8 + (bw / 4 % 2 == 1) * 32;
            //     printf("bit_idx: %d \n", bit_idx);
            //     printf("tree_parent_bit_idx(bit_idx): %d \n", tree_parent_bit_idx(bit_idx));
            //     printf("tree_isset_bit(tree, tree_parent_bit_idx(bit_idx)): %d \n", tree_isset_bit(tree, tree_parent_bit_idx(bit_idx)));
                
            // }
            switch(level)
            {
                case 0:
                    //with padding (inefficient)
                    for(int i = h-1; i<=h+1; ++i)
                    {
                        for(int j = w-1; j<=w+1; ++j)
                        {
                            // Range checks if we are hanging off the matrix
                            if(i >= 0 && i < dense_height)
                            {
                                if(j >= 0 && j < dense_width)
                                {
                                    out_l0[t][c][i][j] = input[t][c][i][j];
                                }
                            }
                        }
                    }
                    break;
                case 1:
                    for(int i = h-1; i<=h+1; ++i)
                    {
                        for(int j = w-1; j<=w+1; ++j)
                        {
                            // Range checks if we are hanging off the matrix
                            if(i >= 0 && i < dense_height)
                            {
                                if(j >= 0 && j < dense_width)
                                {
                                    out_l1[t][c][i][j] = input[t][c][i][j];
                                }
                            }
                        }
                    }
                    break;
                case 2:
                    for(int i = h-1; i<=h+1; ++i)
                    {
                        for(int j = w-1; j<=w+1; ++j)
                        {
                            // Range checks if we are hanging off the matrix
                            if(i >= 0 && i < dense_height)
                            {
                                if(j >= 0 && j < dense_width)
                                {
                                    out_l2[t][c][i][j] = input[t][c][i][j];
                                }
                            }
                        }
                    }
                    break;
                case 3:
                    for(int i = h-1; i<=h+1; ++i)
                    {
                        for(int j = w-1; j<=w+1; ++j)
                        {
                            // Range checks if we are hanging off the matrix
                            if(i >= 0 && i < dense_height)
                            {
                                if(j >= 0 && j < dense_width)
                                {
                                    out_l3[t][c][i][j] = input[t][c][i][j];
                                }
                            }
                        }
                    }
                    break;
                default:
                    break;            
            }

        }

        
    } 


    //TODO: (?)do backward in padded area.
    template <typename scalar_t>
    __global__ void tensor_split_backward_cuda_kernel(
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_in,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out_l0,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out_l1,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out_l2,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out_l3,
        const quadtree stru, float scale_factor_to_grid)
    {

        //batch index
        const int t = blockIdx.z / grad_in.size(1);
        //channel index
        const int c = blockIdx.z - t*grad_in.size(1);
        //height index
        const int h = blockIdx.x * blockDim.x + threadIdx.x;
        //width index
        const int w = blockIdx.y * blockDim.y + threadIdx.y;
        //dense_height
        const int dense_height = grad_in.size(2);
        //dense_width
        const int dense_width = grad_in.size(3);

        if(t < grad_in.size(0) && c < grad_in.size(1) && h < grad_in.size(2) && w < grad_in.size(2))
        {
            //grid index and voxel index inside this grid.
            int grid_idx_h = (h*scale_factor_to_grid);
            int grid_idx_w = (w*scale_factor_to_grid);
            int gw = grid_idx_h >> 3 ;      //  
            int gh = grid_idx_w >> 3;      // grid w h order is differ from tensor
            int bw = grid_idx_h - gw*8;    // int bh = (h*scale_factor) % 8;
            int bh = grid_idx_w - gh*8;    // int bw = (w*scale_factor) % 8;

            int grid_idx =quadtree_grid_idx(&stru, t, gh, gw);
            const qt_tree_t* tree = quadtree_get_tree(&stru, grid_idx);
            int level = tree_level(tree, bh, bw);
            
            switch(level)
            {
                case 0:
                    grad_in[t][c][h][w]=grad_out_l0[t][c][h][w];
                    break;
                case 1:
                    grad_in[t][c][h][w]=grad_out_l1[t][c][h][w];
                    break;
                case 2:
                    grad_in[t][c][h][w]=grad_out_l2[t][c][h][w];
                    break;
                case 3:
                    grad_in[t][c][h][w]=grad_out_l3[t][c][h][w];
                    break;
                default:
                    break;            
            }
        }
    }
}//anonymous namespace


namespace ms
{

    std::vector<torch::Tensor> tensor_split_forward_cuda(
        torch::Tensor input,
        quadtree* stru_ptr)
    {
        auto dim = input.ndimension();
        TORCH_CHECK(dim == 4, "MSError: expected 4D tensor, but got tensor with ", dim, " dimensions instead");
        
        const auto batch_size = input.size(0);
        const auto channel = input.size(1);
        const auto height = input.size(2);
        const auto width = input.size(3);

        TORCH_CHECK(batch_size == stru_ptr->n, "MSError: expected tensors have the same batchsize with structure object");
        TORCH_CHECK(channel == stru_ptr->feature_size, "MSError: expected tensors have the same feature_size with structure object");

        auto out_l0 = torch::zeros_like(input);
        auto out_l1 = torch::zeros_like(input);
        auto out_l2 = torch::zeros_like(input);
        auto out_l3 = torch::zeros_like(input);

        float scale_factor_to_grid = (float)(stru_ptr->grid_height * 8)/height;
        const int threads = 32;
        const dim3 BLOCK_DIM(threads, threads);
        const int blocks = (height + threads - 1) / threads;
        const dim3 GRID_DIM(blocks, blocks, batch_size*channel);

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tensor_split_forward_cuda_kernel", ([&] {
        tensor_split_forward_cuda_kernel<scalar_t><<<GRID_DIM, BLOCK_DIM>>>(
            input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out_l2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out_l3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            *stru_ptr, scale_factor_to_grid);
        }));
        CUDA_POST_KERNEL_CHECK;

        return {out_l0, out_l1, out_l2, out_l3};
    }


    torch::Tensor tensor_split_backward_cuda(
        torch::Tensor grad_out_l0,
        torch::Tensor grad_out_l1,
        torch::Tensor grad_out_l2,
        torch::Tensor grad_out_l3,
        quadtree* stru_ptr)
    {

        auto dim = grad_out_l0.ndimension();
        TORCH_CHECK(dim == 4, "MSError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(grad_out_l0.sizes() == grad_out_l1.sizes(), "MSError: expected src tensors have the same shape");
        TORCH_CHECK(grad_out_l0.sizes() == grad_out_l2.sizes(), "MSError: expected src tensors have the same shape");
        TORCH_CHECK(grad_out_l0.sizes() == grad_out_l3.sizes(), "MSError: expected src tensors have the same shape");

        const auto batch_size = grad_out_l0.size(0);
        const auto channel = grad_out_l0.size(1);
        const auto height = grad_out_l0.size(2);
        const auto width = grad_out_l0.size(3);

        TORCH_CHECK(batch_size == stru_ptr->n, "MSError: expected tensors have the same batchsize with structure object");
        TORCH_CHECK(channel == stru_ptr->feature_size, "MSError: expected tensors have the same feature_size with structure object");

        auto grad_in = torch::zeros_like(grad_out_l0);

        float scale_factor_to_grid = (float)(stru_ptr->grid_height * 8)/height;
        const int threads = 32;
        const dim3 BLOCK_DIM(threads, threads);
        const int blocks = (height + threads - 1) / threads;
        const dim3 GRID_DIM(blocks, blocks, batch_size*channel);

        AT_DISPATCH_FLOATING_TYPES(grad_in.scalar_type(), "tensor_split_backward_cuda_kernel", ([&] {
        tensor_split_backward_cuda_kernel<scalar_t><<<GRID_DIM, BLOCK_DIM>>>(
            grad_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out_l2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out_l3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            *stru_ptr, scale_factor_to_grid);
        }));
        CUDA_POST_KERNEL_CHECK;

        return grad_in;
    } 

} // namespace ms
