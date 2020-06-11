/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-11 16:18:21
 * @ Description: split dense tensor to three sparse tensors with hierarchy of different depths.
 */

 #include <torch/extension.h>
 #include "quadtree.hpp"
 #include "common.hpp"
 
 //回传梯度的时候也要回传边缘信息 在split的时候边缘计算了相应多的次数
 //那么在回传梯度的时候也要把这多的次数加进去 should i ? gradient really need to back to padding?
 //no implement this in this version
 


template <typename scalar_t>
__global__ void DenseCombineForwardKernelGPU(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> in_l0,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> in_l1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> in_l2,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> in_l3,
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
                out[n][f][i][j] = in_l0[n][f][i][j];
                break;
            case 1:
                out[n][f][i][j] = in_l1[n][f][i][j];
                break;
            case 2:
                out[n][f][i][j] = in_l2[n][f][i][j];
                break;
            case 3:
                out[n][f][i][j] = in_l3[n][f][i][j];
                break;
            default;
                break;            
        }
    } // cuda loop

}

template <typename scalar_t>
__global__ void DenseCombineBackwardKernelGPU(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_out,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_in_l0,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_in_l1,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_in_l2,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grid_in_l3,
    const quadtree stru, int N, int feature_size, int dense_height, int dense_width, float scale_factor_to_grid)
{
    for (int vx_idx = blockIdx.x * blockDim.x + threadIdx.x; vx_idx < (N); vx_idx += blockDim.x * gridDim.x)
    {
        //in kernel function / % are inefficient
        //let me just first try this way. then optimize this indexing and then test runing time and do benchmarking.
        //alternative1:
        //          gridDim.x = batchsize
        //          gridDim.y = feature size
        //          gridDim.z = height
        //          blockDim.z = width      (thread)
        //alternative2:
        //          gridDim.x = feature size
        //          gridDim.y = height
        //          gridDim.z = width
        //          blockDim.z = batchsize  (thread)
        //alternative3:
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
    

        int grid_idx =octree_grid_idx(&grid, n, gh, gw);
        const ot_tree_t* tree = octree_get_tree(&grid, grid_idx);
        int level = tree_level(tree, bd, bh, bw);
        switch(level)
        {
        case 0:
            grad_in_l0[n][f][i][j] = grad_out[n][f][i][j];
            break;
        case 1:
            grad_in_l1[n][f][i][j] = grad_out[n][f][i][j];
            break;
        case 2:
            grad_in_l2[n][f][i][j] = grad_out[n][f][i][j];
            break;
        case 3:
            grad_in_l3[n][f][i][j] = grad_out[n][f][i][j];
            break;
        default;
            break;            
        }
    } // cuda loop

}




    
namespace ms
{

    

    void DenseCombineForwardGPU(torch::Tensor output, const torch::Tensor in_l0,
                            const torch::Tensor in_l1, const torch::Tensor in_l2, const torch::Tensor in_l3, const ptr_wrapper<quadtree> stru)
    {
    
        auto dim = input.ndimension();
        auto T = input.size(0);
        auto C = output.size(1);
        auto H = output.size(2);
        auto W = output.size(3);
    
        TORCH_CHECK(dim == 4, "MSError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(output.sizes() == in_l0.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l1.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l2.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l3.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(T == stru->n, "MSError: expected tensors have the same batchsize with structure object");
        TORCH_CHECK(C == stru->feature_size, "MSError: expected tensors have the same feature_size with structure object");
    
        int N = T * C * H * W;
        float scale_factor_to_grid = (float)(stru->grid_height * 8)/H;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "DenseCombineForwardGPU", ([&] {
        DenseCombineForwardKernelGPU<scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
            output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            in_l0.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            in_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            in_l2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            in_l3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            *stru, N, C, H, W, scale_factor_to_grid);
        }));
    } 

    void DenseCombineBackwardGPU(const torch::Tensor grad_out, torch::Tensor grad_in_l0,
        torch::Tensor grad_in_l1, torch::Tensor grad_in_l2, torch::Tensor grad_in_l3, ptr_wrapper<quadtree> stru)
        {
            //make sure out_l* are zero tensors.
            auto dim = grad_out.ndimension();
            auto T = grad_out.size(0);
            auto C = grad_out.size(1);
            auto H = grad_out.size(2);
            auto W = grad_out.size(3);
            
            TORCH_CHECK(dim == 4, "MSError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
            TORCH_CHECK(grad_out.sizes() == grad_in_l0.sizes(), "MSError: expected dst and src tensors have the same shape");
            TORCH_CHECK(grad_out.sizes() == grad_in_l1.sizes(), "MSError: expected dst and src tensors have the same shape");
            TORCH_CHECK(grad_out.sizes() == grad_in_l2.sizes(), "MSError: expected dst and src tensors have the same shape");
            TORCH_CHECK(grad_out.sizes() == grad_in_l3.sizes(), "MSError: expected dst and src tensors have the same shape");
            TORCH_CHECK(T == stru->n, "MSError: expected tensors have the same batchsize with structure object");
            TORCH_CHECK(C == stru->feature_size, "MSError: expected tensors have the same feature_size with structure object");
            
            float scale_factor_to_grid = (float)(stru->grid_height * 8)/H;
            int N = T * C * H * W;
            AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "DenseCombineBackwardGPU", ([&] {
        DenseCombineBackwardGPU<scalar_t><<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
            grad_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_in_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_in_l1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_in_l2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_in_l3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            *stru, N, C, H, W, scale_factor_to_grid);
        }));
    }
} // namespace ms
