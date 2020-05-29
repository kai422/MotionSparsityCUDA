#include <torch/extension.h>

#include "src/utils.hpp"

template<typename Dtype>
void AddCPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c) {
    int N = in_a.numel();
    if (N != in_b.numel())
        throw std::invalid_argument(Formatter()
                                            << "Size mismatch A.numel(): " << in_a.numel()
                                            << ", B.numel(): " << in_b.numel());

    out_c.resize_({N});

    Dtype *a = in_a.data_ptr<Dtype>();
    Dtype *b = in_b.data_ptr<Dtype>();
    Dtype *c = out_c.data_ptr<Dtype>();
    for (int i = 0; i <= N; ++i) {
        c[i] = a[i] + b[i];
    }

}

template void AddCPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);
