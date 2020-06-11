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
__global__ void DenseSplitForwardKernelGPU(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out_l0,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out_l1,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out_l2,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out_l3,
    const quadtree stru, float scale_factor_to_grid)
{
    //batch index
    const int n = blockIdx.z / input.size(1);
    //channel index
    const int c = blockIdx.z - n*input.size(1);
    //height index
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    //width index
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    //dense_height
    const int dense_height = input.size(2)
    //dense_width
    const int dense_width = input.size(3)


    //can change it to swifting <<
    int dense_h = (h*scale_factor_to_grid)
    int dense_w = (w*scale_factor_to_grid)
    int gh = dense_h >> 3;      // int gh = (h*scale_factor_to_grid) / 8;
    int gw = dense_w >> 3;      // int gw = (w*scale_factor_to_grid) / 8;
    int bh = dense_h - gh*8;    // int bh = (h*scale_factor) % 8;
    int bw = dense_w - gw*8;    // int bw = (w*scale_factor) % 8;
        
    int grid_idx =octree_grid_idx(&grid, n, gh, gw);
    const ot_tree_t* tree = octree_get_tree(&grid, grid_idx);
    int level = tree_level(tree, bd, bh, bw);
    switch(level)
    {
        case 0:
            //with padding (inefficient)
            for(int i = h-1, i<=h+1, ++i)
            {
                for(int j = w-1, j<=w+1, ++j)
                {
                    // Range checks if we are hanging off the matrix
                    if(i >= 0 && i < dense_height)
                    {
                        if(j >= 0 && j < dense_width)
                        {
                            out_l0[n][c][i][j] = input[n][c][i][j];
                        }
                    }
                }
            }
            break;
        case 1:
            for(int i = h-1, i<=h+1, ++i)
            {
                for(int j = w-1, j<=w+1, ++j)
                {
                    // Range checks if we are hanging off the matrix
                    if(i >= 0 && i < dense_height)
                    {
                        if(j >= 0 && j < dense_width)
                        {
                            out_l1[n][c][i][j] = input[n][c][i][j];
                        }
                    }
                }
            }
            break;
        case 2:
            for(int i = h-1, i<=h+1, ++i)
            {
                for(int j = w-1, j<=w+1, ++j)
                {
                    // Range checks if we are hanging off the matrix
                    if(i >= 0 && i < dense_height)
                    {
                        if(j >= 0 && j < dense_width)
                        {
                            out_l2[n][c][i][j] = input[n][c][i][j];
                        }
                    }
                }
            }
            break;
        case 3:
            for(int i = h-1, i<=h+1, ++i)
            {
                for(int j = w-1, j<=w+1, ++j)
                {
                    // Range checks if we are hanging off the matrix
                    if(i >= 0 && i < dense_height)
                    {
                        if(j >= 0 && j < dense_width)
                        {
                            out_l3[n][c][i][j] = input[n][c][i][j];
                        }
                    }
                }
            }
            break;
        default;
            break;            
    }
} 


//TODO: ?backward padded area.
template <typename scalar_t>
__global__ void DenseSplitBackwardKernelGPU(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_in,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out_l0,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out_l1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out_l2,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out_l3,
    const quadtree stru, int N, int feature_size, int dense_height, int dense_width, float scale_factor_to_grid)
{
    for (int vx_idx = blockIdx.x * blockDim.x + threadIdx.x; vx_idx < (N); vx_idx += blockDim.x * gridDim.x)
    {
        //in kernel function / % are inefficient
        //let me just first try this way. then optimize this indexing and then test runing time and do benchmarking.
        //TODO:
        //          int THREADS =16
        //          int N = h*w
        //          int BLOCKS = (N+ THREADS-1)/THREADS
        //          dim3 block_dim(THREADS, THREADS)
        //          dim3 grid_dim(BLOCKS, BLOCKS, n)
        //  // Calculate the global thread positions
        //   int row = blockIdx.y * blockDim.y + threadIdx.y;
        //   int col = blockIdx.x * blockDim.x + threadIdx.x;

        int n, d, h, w;
        n = vx_idx / (feature_size * dense_height * dense_width);
        w = vx_idx % dense_width;
        h = ((vx_idx - w) / dense_width) % dense_height;
        f = ((((vx_idx - w) / dense_width) - h) / feature_size) % feature_size;
    

        //can change it to swifting <<
        int gh = (h*scale_factor_to_grid) / 8;
        int gw = (w*scale_factor_to_grid) / 8;
        
        int bh = (h*scale_factor_to_grid) - gh*8;
        int bw = (w*scale_factor_to_grid) - gw*8;
        // int bh = (h*scale_factor) % 8;
        // int bw = (w*scale_factor) % 8;
    

        int grid_idx =octree_grid_idx(&grid, n, gh, gw);
        const ot_tree_t* tree = octree_get_tree(&grid, grid_idx);

        int level = tree_level(tree, bd, bh, bw);
        switch(level)
        {
            case 0:
                grid_in[n][f][i][j]=grid_out_l0[n][f][i][j]
                break;
            case 1:
                grid_in[n][f][i][j]=grid_out_l1[n][f][i][j]
                break;
            case 2:
                grid_in[n][f][i][j]=grid_out_l2[n][f][i][j]
                break;
            case 3:
                grid_in[n][f][i][j]=grid_out_l3[n][f][i][j]
                break;
            default;
                break;            
        }
    } // cuda loop

}
}

namespace ms
{

    std::vector<torch::Tensor> tensor_split_cuda_forward(
        torch::Tensor input,
        ptr_wrapper<quadtree> stru)
    {
        auto dim = input.ndimension();
        TORCH_CHECK(dim == 4, "MSError: expected 4D tensor, but got tensor with ", dim, " dimensions instead");

        const auto batch_size = input.size(0);
        const auto channel = grad_in.size(1);
        const auto height = grad_in.size(2);
        const auto width = grad_in.size(3);

        auto out_l0 = torch::zeros_like(input)
        auto out_l1 = torch::zeros_like(input)
        auto out_l2 = torch::zeros_like(input)
        auto out_l3 = torch::zeros_like(input)

        TORCH_CHECK(batch_size == stru->n, "MSError: expected tensors have the same batchsize with structure object");
        TORCH_CHECK(channel == stru->feature_size, "MSError: expected tensors have the same feature_size with structure object");

        float scale_factor_to_grid = (float)(stru->grid_height * 8)/height;
        const int threads = 32;
        const dim3 BLOCK_DIM(threads, threads)
        const int blocks = (height + threads - 1) / threads;
        const dim3 GRID_DIM(blocks, blocks, batch_size*channel);

        AT_DISPATCH_FLOATING_TYPES(input.type(), "tensor_split_cuda_forward_kernel", ([&] {
        tensor_split_cuda_forward_kernel<scalar_t><<<GRID_DIM, BLOCK_DIM>>>(
            input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out_l2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out_l3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            *stru, scale_factor_to_grid);
        }));

        return {out_l0, out_l1, out_l2, out_l3}
    }


    void DenseSplitBackwardGPU(torch::Tensor grad_in, const torch::Tensor grad_out_l0,
                               const torch::Tensor grad_out_l1, const torch::Tensor grad_out_l2, const torch::Tensor grad_out_l3, const ptr_wrapper<quadtree> stru)
    {

        auto dim = input.ndimension();
        auto T = input.size(0);
        auto C = grad_in.size(1);
        auto H = grad_in.size(2);
        auto W = grad_in.size(3);

        TORCH_CHECK(dim == 4, "MSError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(grad_in.sizes() == grad_out_l0.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_in.sizes() == grad_out_l1.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_in.sizes() == grad_out_l2.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_in.sizes() == grad_out_l3.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(T == stru->n, "MSError: expected tensors have the same batchsize with structure object");
        TORCH_CHECK(C == stru->feature_size, "MSError: expected tensors have the same feature_size with structure object");

        int N = T * C * H * W;
        float scale_factor_to_grid = (float)(stru->grid_height * 8)/H;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "DenseSplitBackwardGPU", ([&] {
        DenseSplitBackwardKernelGPU<scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
            grad_in.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out_l2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out_l3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            *stru, N, C, H, W, scale_factor_to_grid);
        }));
    } 

} // namespace ms
