#include <torch/extension.h>

#include <vector>

#include "common.hpp"
#include "quadtree.hpp"

namespace ms
{

    // CUDA declarations
    quadtree *create_quadtree_structure_cuda(torch::Tensor input);



    // C++ interface
    ptr_wrapper<quadtree> create_quadtree_structure(torch::Tensor input)
    {
        input = input.contiguous();
        CHECK_INPUT(input);

        return create_quadtree_structure_cuda(input);
    }
} // namespace ms