#include <torch/torch.h>
#include <torch/extension.h>

#include "extern.hpp"
#include "quadtree.hpp"

namespace py = pybind11;

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

float array[3] = {3.14, 2.18, -1};

ptr_wrapper<float> get_ptr(void) { return array; }
void use_ptr(ptr_wrapper<float> ptr)
{
    for (int i = 0; i < 3; ++i)
        std::cout << ptr[i] << " ";
    std::cout << "\n";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<ptr_wrapper<float>>(m, "pfloat");
    m.def("get_ptr", &get_ptr);
    m.def("use_ptr", &use_ptr);
    m.def("AddCPU", &AddCPU<float>);
}

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "pybind11 example plugin";
    m.def("AddCPU", &AddCPU<float>);
    //m.def("CreateFromDense", &ms::CreateFromDense, return_value_policy::reference);

    std::string quad_name = std::string("QuadTreeStru");
    py::class_<ms::quadtree>(m, quad_name.c_str())
        .def(py::init<int, int, int>())
        .def("num_block", &ms::quadtree::num_blocks);
}
*/