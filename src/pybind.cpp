#include <torch/torch.h>
#include <torch/extension.h>

#include "extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "pybind11 example plugin";
    m.def("AddCPU", &AddCPU<float>, py::return_value_policy::reference);
    //m.def("CreateFromDense", &ms::CreateFromDense, return_value_policy::reference);
}
