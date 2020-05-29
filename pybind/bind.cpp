#include <torch/torch.h>
#include <torch/extension.h>

#include "pybind/extern.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("AddCPU", &AddCPU<float>);
}
