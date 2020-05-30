#include <torch/extension.h>
#include <iostream>

template <typename Dtype>
int *AddCPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c)
{
    int N = in_a.numel();
    if (N != in_b.numel())
        throw std::invalid_argument("Size mismatch");

    out_c.resize_({N});
    std::cout << "calling AddCPU_cpp" << std::endl;
    Dtype *a = in_a.data_ptr<Dtype>();
    Dtype *b = in_b.data_ptr<Dtype>();
    Dtype *c = out_c.data_ptr<Dtype>();
    for (int i = 0; i <= N; ++i)
    {
        c[i] = a[i] + b[i];
    }
    int *array = new int[2]{};
    std::cout << array << std::endl;
    return array;
}

template int *AddCPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("AddCPU", &AddCPU<float>);
}
