/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-06-01 20:50:57
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-07 23:24:00
 * @ Description:
 */

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
    m.def("quadtree_resize_fsize_batch", &ms::quadtree_resize_fsize_batch);
    m.def("SaveQuadStruAsImgCPU", &ms::SaveQuadStruAsImg);
}
