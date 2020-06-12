#include <torch/torch.h>
#include <torch/extension.h>

#include "extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<ptr_wrapper<quadtree>>(m, "ptr_quad");

    m.def("create_quadtree_structure", &ms::create_quadtree_structure, "create_quadtree_structure(CUDA)");
    m.def("tensor_split_forward", &ms::tensor_split_forward, "tensor_split_forward(CUDA)");
    m.def("tensor_split_backward", &ms::tensor_split_backward, "tensor_split_backwar(CUDA)");
    // m.def("DenseSplitBackwardCPU", &ms::DenseSplitBackwardCPU);
    // m.def("DenseCombineForwardCPU", &ms::DenseCombineForwardCPU);
    // m.def("DenseCombineBackwardCPU", &ms::DenseCombineBackwardCPU);
    // m.def("quadtree_pool2x2_stru_batch", &ms::quadtree_pool2x2_stru_batch);
    // m.def("SaveQuadStruAsImg", &ms::SaveQuadStruAsImg);
}
