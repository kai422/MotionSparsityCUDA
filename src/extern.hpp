template <typename Dtype>
int *AddCPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);

namespace ms
{
    quadtree **CreateFromDense(at::Tensor &input);

}