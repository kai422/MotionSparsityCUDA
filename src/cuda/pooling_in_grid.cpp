#include <torch/extension.h>

#include <vector>

#include "common.hpp"
#include "quadtree.hpp"

namespace ms
{
    // CUDA declarations
    torch::Tensor pooling_in_grid_cuda(torch::Tensor input, quadtree *stru);



    // C++ interface
    torch::Tensor pooling_in_grid(torch::Tensor input, ptr_wrapper<quadtree> stru)
    {
        input = input.contiguous();
        CHECK_INPUT(input);

        return pooling_in_grid_cuda(input, stru.get());
    }
} // namespace ms
