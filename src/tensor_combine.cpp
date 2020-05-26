/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-26 11:57:43
 * @ Description: combine sparse tensors with hierarchy of different depths.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quad2dense.hpp"
#include "quadtree.hpp"

namespace ms
{

    //intput Tensor in1 with only first layer.
    //intput Tensor in2 with only second layer.
    //intput Tensor in3 with only third layer.
    //intput Tensor in4 with only fourth layer.

    //out Tensor out
    //template <typename Dtype>
    void DenseCombineForwardCPU(at::Tensor &in_l1_r, at::Tensor &in_l2_r, at::Tensor &in_l3_r, at::Tensor &in_l4_r, at::Tensor &output_r, quadtree *structures[])
    {
        auto in_l1 = in_l1_r;
        auto in_l2 = in_l2_r;
        auto in_l3 = in_l3_r;
        auto in_l4 = in_l4_r;
        auto output = output_r;

        auto dim = output.ndimension();

        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(output.sizes() == in_l1.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l2.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l3.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l4.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");

        in_l1 = in_l1.contiguous();
        in_l2 = in_l2.contiguous();
        in_l3 = in_l3.contiguous();
        in_l4 = in_l4.contiguous();
        output = output.contiguous();

        auto T = output.size(0);
        auto f = output.size(1);
        auto h = output.size(2);
        auto w = output.size(3);
        at::parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
            for (auto t = start; t < end; t++)
            {
                auto stru_t = structures[t];
                auto in_l1_t = in_l1[t];
                auto in_l2_t = in_l2[t];
                auto in_l3_t = in_l3[t];
                auto in_l4_t = in_l4[t];
                auto output_t = output[t];

                //combine from three dense tensor to one gridtree
                quadtree *output_quad;
                output_quad = CombineDensetToQuad(f, h, w, output_quad, stru_t, in_l1_t.data_ptr<float>(), in_l2_t.data_ptr<float>(), in_l3_t.data_ptr<float>(), in_l4_t.data_ptr<float>());

                QuadToDense(output_quad, f, h, w, output_t.data_ptr<float>());

                //convert gridtree to dense tensor
            }
        });
    }

    quadtree *CombineDensetToQuad(const int &f, const int &tensor_h, const int &tensor_w, quadtree *output_quad, quadtree *stru, float *int_l1_dst, float *int_l2_dst, float *int_l3_dst, float *int_l4_dst)
    {
    }

    void DenseCombineBackwardCPU(at::Tensor &in_l1_r, at::Tensor &in_l2_r, at::Tensor &in_l3_r, at::Tensor &in_l4_r, at::Tensor &output_r, quadtree *structures[])
    {
    }

} // namespace ms
