#include <torch/torch.h>
#include <torch/extension.h>

#include "extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<ptr_wrapper<ms::quadtree *>>(m, "ptr_ptr_quad");
    m.def("CreateFromDense", &ms::CreateFromDense);
}
