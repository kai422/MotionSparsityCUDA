/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-22 18:48:09
 * @ Description: split dense tensor to three sparse tensors with hierarchy of different depths.
 */

#include <pybind11/pybind11.h>
#include "commdef.hpp"
#include "quadtree.hpp"

namespace ms
{

    //input Tensor in
    //output Tensor out1 with only first layer.
    //output Tensor out2 with only second layer.
    //output Tensor out3 with only third layer.
    template <typename Dtype>
    void DenseSplitForwardCPU(at::Tensor &input_r, at::Tensor &out1,
                              at::Tensor &out2, at::Tensor &out3, const quadtree &stru)
    {
        auto input = input_r;
        auto dim = input.ndimension();
        c10::IntArrayRef input_sizes = weight.sizes();
        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        input = input.contiguous();

        auto T = input.size(0);
        auto f = input.size(1);
        auto h = input.size(2);
        auto w = input.size(3);
        at::parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
            for (auto t = start; t < end; t++)
            {
                auto input_t = input[t];
                //create from dense
                quadtree *in_tree;
                in_tree = DenseToQuad(f, h, w, input_t.template data_ptr<Dtype>(), stru);

                //split
                quadtree *out_tree_1;
                quadtree *out_tree_2;
                quadtree *out_tree_3;
                intree.split(out_tree_1, out_tree_2, out_tree_3);

                //quad to dense
                out1 = out_tree_1.toDense();
                out2 = out_tree_2.toDense();
                out3 = out_tree_3.toDense();
            }
        
        }
    }

    //TODO:pooling in the grid.
    //TODO:grid pooling to adapt grid tree size.

} //end namespace ms.