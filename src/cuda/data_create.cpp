#include <torch/extension.h>

#include <vector>

#include "common.hpp"
#include "quadtree.hpp"
// CUDA declarations

quadtree *create_quadtree_structure_cuda(torch::Tensor input);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

ptr_wrapper<quadtree> create_quadtree_structure(torch::Tensor input)
{
    input = input.contiguous();
    CHECK_INPUT(input);

    return create_quadtree_structure_cuda(input);
}