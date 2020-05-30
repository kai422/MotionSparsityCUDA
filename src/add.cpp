#include <torch/extension.h>
#include <iostream>

template <class T>
class ptr_wrapper
{
public:
    ptr_wrapper() : ptr(nullptr) {}
    ptr_wrapper(T *ptr) : ptr(ptr) {}
    ptr_wrapper(const ptr_wrapper &other) : ptr(other.ptr) {}
    T &operator*() const { return *ptr; }
    T *operator->() const { return ptr; }
    T *get() const { return ptr; }
    void destroy() { delete ptr; }
    T &operator[](std::size_t idx) const { return ptr[idx]; }

private:
    T *ptr;
};

template <typename Dtype>
ptr_wrapper<Dtype> AddCPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c)
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
    Dtype *array = new Dtype[2]{1, 2};
    std::cout << array << std::endl;
    return array;
}

template <typename Dtype>
void printCPU(ptr_wrapper<Dtype> ptr)
{
    for (int i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    {
        std::cout << ptr[i] << std::endl;
    }
}

template void printCPU<float>(ptr_wrapper<float> ptr);
template ptr_wrapper<float> AddCPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<ptr_wrapper<float>>(m, "pfloat");
    m.def("AddCPU", &AddCPU<float>);
    m.def("printCPU", &printCPU<float>);
}
