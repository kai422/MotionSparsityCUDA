/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-11 16:18:21
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
    __global__ void tensor_combine_forward_cuda_kernel(
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input_l0,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input_l1,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input_l2,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input_l3,
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

        //grid index and voxel index inside this grid.
        int dense_h = (h*scale_factor_to_grid);
        int dense_w = (w*scale_factor_to_grid);
        int gh = dense_h >> 3;      // int gh = (h*scale_factor_to_grid) / 8;
        int gw = dense_w >> 3;      // int gw = (w*scale_factor_to_grid) / 8;
        int bh = dense_h - gh*8;    // int bh = (h*scale_factor) % 8;
        int bw = dense_w - gw*8;    // int bw = (w*scale_factor) % 8;
    

        int grid_idx =quadtree_grid_idx(&stru, n, gh, gw);
        const qt_tree_t* tree = quadtree_get_tree(&stru, grid_idx);
        int level = tree_level(tree, bh, bw);

        switch(level)
        {
            case 0:
                output[t][c][h][w] = input_l0[t][c][h][w];
                break;
            case 1:
                output[t][c][h][w] = input_l1[t][c][h][w];
                break;
            case 2:
                output[t][c][h][w] = input_l2[t][c][h][w];
                break;
            case 3:
                output[t][c][h][w] = input_l3[t][c][h][w];
                break;
            default;
                break;            
        }
    }


    template <typename scalar_t>
    __global__ void tensor_combine_backward_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_out,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_in_l0,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_in_l1,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_in_l2,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_in_l3,
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

        //grid index and voxel index inside this grid.
        int dense_h = (h*scale_factor_to_grid);
        int dense_w = (w*scale_factor_to_grid);
        int gh = dense_h >> 3;      // int gh = (h*scale_factor_to_grid) / 8;
        int gw = dense_w >> 3;      // int gw = (w*scale_factor_to_grid) / 8;
        int bh = dense_h - gh*8;    // int bh = (h*scale_factor) % 8;
        int bw = dense_w - gw*8;    // int bw = (w*scale_factor) % 8;

        int grid_idx =quadtree_grid_idx(&stru, t, gh, gw);
        const qt_tree_t* tree = quadtree_get_tree(&stru, grid_idx);
        int level = tree_level(tree, bh, bw);
        switch(level)
        {
            case 0:
                grad_in_l0[t][c][h][w] = grad_out[t][c][h][w];
                break;
            case 1:
                grad_in_l1[t][c][h][w] = grad_out[t][c][h][w];
                break;
            case 2:
                grad_in_l2[t][c][h][w] = grad_out[t][c][h][w];
                break;
            case 3:
                grad_in_l3[t][c][h][w] = grad_out[t][c][h][w];
                break;
            default;
                break;            
        }
    } 

}//anonymous namespace








    
namespace ms
{

    torch::Tensor tensor_combine_forward_cuda(
        torch::Tensor input_l0,
        torch::Tensor input_l1,
        torch::Tensor input_l2,
        torch::Tensor input_l3,
        quadtree *stru_ptr)
    {
        auto dim = input_l0.ndimension();
        TORCH_CHECK(dim == 4, "MSError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(input_l0.sizes() == input_l1.sizes(), "MSError: expected src tensors have the same shape");
        TORCH_CHECK(input_l0.sizes() == input_l2.sizes(), "MSError: expected src tensors have the same shape");
        TORCH_CHECK(input_l0.sizes() == input_l3.sizes(), "MSError: expected src tensors have the same shape");

        const auto batch_size = input_l0.size(0);
        const auto channel = input_l0.size(1);
        const auto height = input_l0.size(2);
        const auto width = input_l0.size(3);
    
        TORCH_CHECK(batch_size == stru_ptr->n, "MSError: expected tensors have the same batchsize with structure object");
        TORCH_CHECK(channel == stru_ptr->feature_size, "MSError: expected tensors have the same feature_size with structure object");

        auto output = torch::zeros_like(input_l0);
    
        float scale_factor_to_grid = (float)(stru_ptr->grid_height * 8)/height;
        const int threads = 32;
        const dim3 BLOCK_DIM(threads, threads);
        const int blocks = (height + threads - 1) / threads;
        const dim3 GRID_DIM(blocks, blocks, batch_size*channel);

        AT_DISPATCH_FLOATING_TYPES(input_l0.scalar_type(), "tensor_combine_forward_cuda_kernel", ([&] {
        tensor_combine_forward_cuda_kernel<scalar_t><<<GRID_DIM, BLOCK_DIM>>>(
            output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input_l2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input_l3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            *stru, scale_factor_to_grid);
        }));
        CUDA_POST_KERNEL_CHECK;
        return output;
    } 

    std::vector<torch::Tensor> tensor_combine_backward_cuda(
        torch::Tensor grad_out,
        quadtree *stru_ptr);
    {
        auto dim = grad_out.ndimension();
        TORCH_CHECK(dim == 4, "MSError: expected 4D tensor, but got tensor with ", dim, " dimensions instead");
        const auto batch_size = grad_out.size(0);
        const auto channel = grad_out.size(1);
        const auto height = grad_out.size(2);
        const auto width = grad_out.size(3);
        
        TORCH_CHECK(batch_size == stru_ptr->n, "MSError: expected tensors have the same batchsize with structure object");
        TORCH_CHECK(channel == stru_ptr->feature_size, "MSError: expected tensors have the same feature_size with structure object");

        auto grad_in_l0 = torch::zeros_like(input);
        auto grad_in_l1 = torch::zeros_like(input);
        auto grad_in_l2 = torch::zeros_like(input);
        auto grad_in_l3 = torch::zeros_like(input);
        
        float scale_factor_to_grid = (float)(stru_ptr->grid_height * 8)/height;
        const int threads = 32;
        const dim3 BLOCK_DIM(threads, threads);
        const int blocks = (height + threads - 1) / threads;
        const dim3 GRID_DIM(blocks, blocks, batch_size*channel);

        AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "tensor_combine_backward_cuda_kernel", ([&] {
        tensor_combine_backward_cuda_kernel<scalar_t><<<GRID_DIM, BLOCK_DIM>>>(
            grad_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_in_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_in_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_in_l2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_in_l3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            *stru_ptr, scale_factor_to_grid);
        }));
        CUDA_POST_KERNEL_CHECK;
        return {grad_in_l0, grad_in_l1, grad_in_l2, grad_in_l3};
    }
} // namespace ms
