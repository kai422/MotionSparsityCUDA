/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-19 22:12:18
 * @ Description: split dense tensor to three spot tensors with hierarchy of different depths.
 */

#include <pybind11/pybind11.h>
#include "commdef.hpp"
#include "quadtree.hpp"

namespace ms
{

    template <typename Dtype>
    void DenseSplitForwardCPU(at::Tensor in, at::Tensor out1,
                              at::Tensor out2, at::Tensor out3, quadtree grid)
    {
        assert(in.size(0) == out1.size(0) && "size mismatch");
        assert(in.size(1) == out1.size(1) && "size mismatch");
        assert(in.size(2) == out1.size(2) && "size mismatch");
    }
} //end namespace ms.