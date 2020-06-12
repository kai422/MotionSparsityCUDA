#include <torch/torch.h>
#include <torch/extension.h>

#include "extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<ptr_wrapper<quadtree>>(m, "ptr_quad");

    m.def("create_quadtree_structure", &ms::create_quadtree_structure, "create_quadtree_structure(CUDA)");
    //m.def("tensor_split_forward", &ms::tensor_split_forward, "tensor_split_forward(CUDA)");
    //m.def("tensor_split_backward", &ms::tensor_split_backward, "tensor_split_backward(CUDA)");
    //m.def("tensor_combine_forward", &ms::tensor_combine_forward, "tensor_combine_forward(CUDA)");
    //m.def("tensor_combine_backward", &ms::tensor_combine_backward, "tensor_combine_backward(CUDA)");
    //m.def("pooling_in_grid", &ms::pooling_in_grid, "pooling_in_grid(CUDA)");

    // m.def("quadtree_pool2x2_stru_batch", &ms::quadtree_pool2x2_stru_batch);
    m.def("SaveQuadStruAsImg", &ms::SaveQuadStruAsImg);
}
