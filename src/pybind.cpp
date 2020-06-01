#include <torch/torch.h>
#include <torch/extension.h>

#include "extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<ptr_wrapper<ms::quadtree *>>(m, "ptr_ptr_quad");
    m.def("CreateFromDense", &ms::CreateFromDense);
    m.def("DenseSplitForwardCPU", &ms::DenseSplitForwardCPU);
    m.def("DenseSplitBackwardCPU", &ms::DenseSplitBackwardCPU);
    m.def("DenseCombineForwardCPU", &ms::DenseCombineForwardCPU);
    m.def("DenseCombineBackwardCPU", &ms::DenseCombineBackwardCPU);
    m.def("quadtree_pool2x2_stru_batch", &ms::quadtree_pool2x2_stru_batch);
}
