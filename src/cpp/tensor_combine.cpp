/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-08 23:56:56
 * @ Description: combine sparse tensors with hierarchy of different depths.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quadtodense.hpp"
#include "quadtree.hpp"
#include "tensor_common.hpp"
#include "common.hpp"
#include "tensor_combine.hpp"

namespace ms
{

    //intput Tensor in1 with only first layer.
    //intput Tensor in2 with only second layer.
    //intput Tensor in3 with only third layer.
    //intput Tensor in4 with only fourth layer.

    //out Tensor out
    //template <typename Dtype>
    void DenseCombineForwardCPU(torch::Tensor in_l1, torch::Tensor in_l2, torch::Tensor in_l3, torch::Tensor in_l4, torch::Tensor output, ptr_wrapper<quadtree *> structures)
    {
        auto dim = output.ndimension();

        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(output.sizes() == in_l1.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l2.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l3.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(output.sizes() == in_l4.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");

        auto T = output.size(0);
        auto f = output.size(1);
        auto h = output.size(2);
        auto w = output.size(3);
        parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
            for (auto t = start; t < end; t++)
            {
                auto stru_t = structures[t];
                auto in_l1_t = in_l1[t];
                auto in_l2_t = in_l2[t];
                auto in_l3_t = in_l3[t];
                auto in_l4_t = in_l4[t];
                auto output_t = output[t];

                //combine from three dense tensor to one gridtree
                CombineDenseToQuad(f, h, w, stru_t, in_l1_t, in_l2_t, in_l3_t, in_l4_t);

                QuadToDense(stru_t, f, h, w, output_t);

                //convert gridtree to dense tensor
            }
        });
    }

    quadtree *CombineDenseToQuad(const int &f, const int &tensor_h, const int &tensor_w, quadtree *stru, torch::Tensor in_l1_src, torch::Tensor in_l2_src, torch::Tensor in_l3_src, torch::Tensor in_l4_src)
    {
        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index
        // tensor_size tensor_h x tensor_w (256x256)
        // grid size 64x64
        // each grid at most 8x8 leaves(at current mv resolution)
        // each leaf has 8x8 pixels
        assert(f == stru->feature_size && ((float)tensor_h / stru->grid_height) == ((float)stru->grid_width / tensor_w) &&
               "expect input structure has same size with data tensor.");
        float scale_factor = (float)tensor_h / (stru->grid_height * 8);

        quadtree *output = stru;
        if (output->data != nullptr)
        {
            delete[] output->data;
        }
        output->data = new qt_data_t[output->n_leafs * output->feature_size]{};

        int n_blocks = output->num_blocks();
        int grid_width = output->grid_width;
        int feature_size = output->feature_size;
#pragma omp parallel for
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            bitset<21UL> &grid_tree = output->trees[grid_idx];
            qt_data_t *grid_data = output->data + output->feature_size * output->prefix_leafs[grid_idx];

            int grid_h_idx = grid_idx / grid_width;
            int grid_w_idx = grid_idx % grid_width;
            float centre_x = grid_w_idx * 8 + 4;
            float centre_y = grid_h_idx * 8 + 4;

            if (tree_isset_bit(grid_tree, 0))
            {
                for (int hl1 = 0; hl1 < 2; ++hl1)
                {
                    for (int wl1 = 0; wl1 < 2; ++wl1)
                    {
                        int bit_idx_l1 = 1 + hl1 * 2 + wl1;
                        float centre_x_l1 = centre_x + (wl1 * 4) - 2;
                        float centre_y_l1 = centre_y + (hl1 * 4) - 2;
                        if (tree_isset_bit(grid_tree, bit_idx_l1))
                        {
                            for (int hl2 = 0; hl2 < 2; ++hl2)
                            {
                                for (int wl2 = 0; wl2 < 2; ++wl2)
                                {
                                    int bit_idx_l2 = child_idx(bit_idx_l1) + hl2 * 2 + wl2;
                                    float centre_x_l2 = centre_x_l1 + (wl2 * 2) - 1;
                                    float centre_y_l2 = centre_y_l1 + (hl2 * 2) - 1;
                                    if (tree_isset_bit(grid_tree, bit_idx_l2))
                                    {
                                        for (int hl3 = 0; hl3 < 2; ++hl3)
                                        {
                                            for (int wl3 = 0; wl3 < 2; ++wl3)
                                            {
                                                int bit_idx_l3 = child_idx(bit_idx_l2) + hl3 * 2 + wl3;
                                                float centre_x_l3 = centre_x_l2 + (wl3 * 1) - 0.5;
                                                float centre_y_l3 = centre_y_l2 + (hl3 * 1) - 0.5;
                                                int data_idx = tree_data_idx(grid_tree, bit_idx_l3, feature_size);
                                                get_data_from_tensor(grid_data + data_idx, in_l4_src, scale_factor, feature_size, centre_x_l3 - 0.5, centre_x_l3 + 0.5, centre_y_l3 - 0.5, centre_y_l3 + 0.5);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        int data_idx = tree_data_idx(grid_tree, bit_idx_l2, feature_size);
                                        get_data_from_tensor(grid_data + data_idx, in_l3_src, scale_factor, feature_size, centre_x_l2 - 1, centre_x_l2 + 1, centre_y_l2 - 1, centre_y_l2 + 1);
                                    }
                                }
                            }
                        }
                        else
                        {
                            int data_idx = tree_data_idx(grid_tree, bit_idx_l1, feature_size);
                            get_data_from_tensor(grid_data + data_idx, in_l2_src, scale_factor, feature_size, centre_x_l1 - 2, centre_x_l1 + 2, centre_y_l1 - 2, centre_y_l1 + 2);
                        }
                    }
                }
            }
            else
            {
                //if not set, average the content

                get_data_from_tensor(grid_data, in_l1_src, scale_factor, feature_size, centre_x - 4, centre_x + 4, centre_y - 4, centre_y + 4);
            }
        }

        return output;
    }

    void DenseCombineBackwardCPU(torch::Tensor grad_in_l1, torch::Tensor grad_in_l2, torch::Tensor grad_in_l3, torch::Tensor grad_in_l4, torch::Tensor grad_out, ptr_wrapper<quadtree *> structures)
    {
        auto dim = grad_out.ndimension();

        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(grad_out.sizes() == grad_in_l1.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_out.sizes() == grad_in_l2.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_out.sizes() == grad_in_l3.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_out.sizes() == grad_in_l4.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");

        auto T = grad_out.size(0);
        auto f = grad_out.size(1);
        auto h = grad_out.size(2);
        auto w = grad_out.size(3);
        parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
            for (auto t = start; t < end; t++)
            {
                auto stru_t = structures[t];
                auto grad_in_l1_t = grad_in_l1[t];
                auto grad_in_l2_t = grad_in_l2[t];
                auto grad_in_l3_t = grad_in_l3[t];
                auto grad_in_l4_t = grad_in_l4[t];
                auto grad_out_t = grad_out[t];

                // data_ptr accessor: f_index*(h*w) + h_index*w + w_index
                // tensor_size tensor_h x tensor_w (256x256)
                // grid size 64x64
                // each grid at most 8x8 leaves(at current mv resolution)
                // each leaf has 8x8 pixels
                assert(f == stru_t->feature_size && ((float)h / stru_t->grid_height) == ((float)stru_t->grid_width / w) &&
                       "expect input structure has same size with data tensor.");
                _unused(f);
                _unused(w);
                float scale_factor = (float)h / (stru_t->grid_height * 8);

                int n_blocks = stru_t->num_blocks();
                int grid_width = stru_t->grid_width;
                int feature_size = stru_t->feature_size;
#pragma omp parallel for
                for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
                {
                    bitset<21UL> &grid_tree = stru_t->trees[grid_idx];

                    int grid_h_idx = grid_idx / grid_width;
                    int grid_w_idx = grid_idx % grid_width;
                    float centre_x = grid_w_idx * 8 + 4;
                    float centre_y = grid_h_idx * 8 + 4;

                    if (tree_isset_bit(grid_tree, 0))
                    {
                        for (int hl1 = 0; hl1 < 2; ++hl1)
                        {
                            for (int wl1 = 0; wl1 < 2; ++wl1)
                            {
                                int bit_idx_l1 = 1 + hl1 * 2 + wl1;
                                float centre_x_l1 = centre_x + (wl1 * 4) - 2;
                                float centre_y_l1 = centre_y + (hl1 * 4) - 2;
                                if (tree_isset_bit(grid_tree, bit_idx_l1))
                                {
                                    for (int hl2 = 0; hl2 < 2; ++hl2)
                                    {
                                        for (int wl2 = 0; wl2 < 2; ++wl2)
                                        {
                                            int bit_idx_l2 = child_idx(bit_idx_l1) + hl2 * 2 + wl2;
                                            float centre_x_l2 = centre_x_l1 + (wl2 * 2) - 1;
                                            float centre_y_l2 = centre_y_l1 + (hl2 * 2) - 1;
                                            if (tree_isset_bit(grid_tree, bit_idx_l2))
                                            {
                                                for (int hl3 = 0; hl3 < 2; ++hl3)
                                                {
                                                    for (int wl3 = 0; wl3 < 2; ++wl3)
                                                    {
                                                        float centre_x_l3 = centre_x_l2 + (wl3 * 1) - 0.5;
                                                        float centre_y_l3 = centre_y_l2 + (hl3 * 1) - 0.5;
                                                        assign_data_among_tensor(grad_in_l4_t, grad_out_t, scale_factor, feature_size, centre_x_l3 - 0.5, centre_x_l3 + 0.5, centre_y_l3 - 0.5, centre_y_l3 + 0.5);
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                assign_data_among_tensor(grad_in_l3_t, grad_out_t, scale_factor, feature_size, centre_x_l2 - 1, centre_x_l2 + 1, centre_y_l2 - 1, centre_y_l2 + 1);
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    assign_data_among_tensor(grad_in_l2_t, grad_out_t, scale_factor, feature_size, centre_x_l1 - 2, centre_x_l1 + 2, centre_y_l1 - 2, centre_y_l1 + 2);
                                }
                            }
                        }
                    }
                    else
                    {
                        //if not set, average the content
                        assign_data_among_tensor(grad_in_l1_t, grad_out_t, scale_factor, feature_size, centre_x - 4, centre_x + 4, centre_y - 4, centre_y + 4);
                    }
                }
            }
        });
    }

} // namespace ms
