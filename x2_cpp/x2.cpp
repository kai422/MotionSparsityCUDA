#include <torch/extension.h>
#include <iostream>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>

torch::Tensor x2(torch::Tensor z)
{
    torch::Tensor z_out = at::empty({z.size(0), z.size(1)}, z.options());
    int64_t batch_size = z.size(0);

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++)
        {
            z_out[b] = z[b] * z[b];
        }
        std::cout << "hi there from " << omp_get_thread_num() << std::endl;
    });

    return z_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("x2", &x2, "square");
}