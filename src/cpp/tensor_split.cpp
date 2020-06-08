/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-08 23:30:49
 * @ Description: split dense tensor to three sparse tensors with hierarchy of different depths.
 */

// /*******************breakpoint******************/
// for (int i = 0; i < stru_t->n_leafs; ++i)
// {
//     std::cout << stru_t->data[i] << ' ';
// }

// TORCH_CHECK(1 == 0, "MotionSparsityError: breakpoint0");
/*******************breakpoint******************/

#include <torch/extension.h>
#include "quadtree.hpp"
#include "densetoquad.hpp"
#include "common.hpp"
#include "tensor_common.hpp"
#include "tensor_split.hpp"

namespace ms
{

    void DenseSplitForwardCPU(const torch::Tensor input, torch::Tensor out_l1,
                              torch::Tensor out_l2, torch::Tensor out_l3, torch::Tensor out_l4, ptr_wrapper<quadtree *> structures)
    {
        //please make sure out_l_r are zero tensor.

        auto dim = input.ndimension();

        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(input.sizes() == out_l1.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(input.sizes() == out_l2.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(input.sizes() == out_l3.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(input.sizes() == out_l4.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");

        auto T = input.size(0);
        auto f = input.size(1);
        auto h = input.size(2);
        auto w = input.size(3);
        parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
            for (auto t = start; t < end; t++)
            {
                auto stru_t = structures[t];
                auto input_t = input[t];
                auto out_l1_t = out_l1[t];
                auto out_l2_t = out_l2[t];
                auto out_l3_t = out_l3[t];
                auto out_l4_t = out_l4[t];

                //create from dense
                DenseToQuad(f, h, w, input_t, stru_t);

                //split to three tensor with padding
                std::vector<std::tuple<int, int>> border_coords_l1;
                std::vector<std::tuple<int, int>> border_coords_l2;
                std::vector<std::tuple<int, int>> border_coords_l3;
                std::vector<std::tuple<int, int>> border_coords_l4;

                splitQuadToDense(f, h, w, stru_t, out_l1_t, out_l2_t, out_l3_t, out_l4_t, border_coords_l1, border_coords_l2, border_coords_l3, border_coords_l4);

                get_padded_tensor(out_l1_t, input_t, border_coords_l1);
                get_padded_tensor(out_l2_t, input_t, border_coords_l2);
                get_padded_tensor(out_l3_t, input_t, border_coords_l3);
                get_padded_tensor(out_l4_t, input_t, border_coords_l4);
            }
        });
        //too slow
        //try parallel
    }

    void splitQuadToDense(const int &f, const int &tensor_h, const int &tensor_w, quadtree *input_quad, torch::Tensor out_l1_dst, torch::Tensor out_l2_dst, torch::Tensor out_l3_dst, torch::Tensor out_l4_dst, std::vector<std::tuple<int, int>> &border_coords_l1, std::vector<std::tuple<int, int>> &border_coords_l2, std::vector<std::tuple<int, int>> &border_coords_l3, std::vector<std::tuple<int, int>> &border_coords_l4)
    {
        int n_blocks = input_quad->num_blocks();
        int grid_height = input_quad->grid_height;
        int grid_width = input_quad->grid_width;
        int feature_size = input_quad->feature_size;

        assert(f == feature_size && ((float)tensor_h / input_quad->grid_height) == ((float)input_quad->grid_width / tensor_w) &&
               "expect input structure has same size with data tensor.");
        float scale_factor = (float)tensor_h / (grid_height * 8);
#pragma omp parallel for
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            bitset<21UL> &grid_tree = input_quad->trees[grid_idx];
            qt_data_t *grid_data = input_quad->data + input_quad->feature_size * input_quad->prefix_leafs[grid_idx];

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
                                                save_data_to_tensor_with_border_coords(grid_data + data_idx, out_l4_dst, scale_factor, feature_size, centre_x_l3 - 0.5, centre_x_l3 + 0.5, centre_y_l3 - 0.5, centre_y_l3 + 0.5, border_coords_l4);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        int data_idx = tree_data_idx(grid_tree, bit_idx_l2, feature_size);
                                        save_data_to_tensor_with_border_coords(grid_data + data_idx, out_l3_dst, scale_factor, feature_size, centre_x_l2 - 1, centre_x_l2 + 1, centre_y_l2 - 1, centre_y_l2 + 1, border_coords_l3);
                                    }
                                }
                            }
                        }
                        else
                        {
                            int data_idx = tree_data_idx(grid_tree, bit_idx_l1, feature_size);
                            save_data_to_tensor_with_border_coords(grid_data + data_idx, out_l2_dst, scale_factor, feature_size, centre_x_l1 - 2, centre_x_l1 + 2, centre_y_l1 - 2, centre_y_l1 + 2, border_coords_l2);
                        }
                    }
                }
            }
            else
            {
                //ouput whole grid(cx-4,cx+4,cy-4,cy+4) to out_l1_dst tensor

                save_data_to_tensor_with_border_coords(grid_data, out_l1_dst, scale_factor, feature_size, centre_x - 4, centre_x + 4, centre_y - 4, centre_y + 4, border_coords_l1);
            }
        }
    }

    void get_padded_tensor(torch::Tensor input_tensor, torch::Tensor ref, std::vector<std::tuple<int, int>> &border_coords)
    {
        //if the position in the input != 0;
        //then check its neighbor
        //it == 0
        //then pad it (i.e. get the value from the ref tensor)
        int feature_size = input_tensor.size(0);
        int h;
        int w;
        for (auto cor : border_coords)
        {
            for (int f = 0; f < feature_size; ++f)
            {
                h = std::get<0>(cor);
                w = std::get<1>(cor);
                input_tensor[f][h][w] = ref[f][h][w];
            }
        }
    }

    void DenseSplitBackwardCPU(torch::Tensor grad_in, torch::Tensor grad_out_l1,
                               torch::Tensor grad_out_l2, torch::Tensor grad_out_l3, torch::Tensor grad_out_l4, ptr_wrapper<quadtree *> structures)
    {

        auto dim = grad_in.ndimension();

        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        TORCH_CHECK(grad_in.sizes() == grad_out_l1.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_in.sizes() == grad_out_l2.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_in.sizes() == grad_out_l3.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");
        TORCH_CHECK(grad_in.sizes() == grad_out_l4.sizes(), "MotionSparsityError: expected dst and src tensors have the same shape");

        auto T = grad_in.size(0);
        auto f = grad_in.size(1);
        auto h = grad_in.size(2);
        auto w = grad_in.size(3);
        parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
            for (auto t = start; t < end; t++)
            {
                auto stru_t = structures[t];
                auto grad_out_l1_t = grad_out_l1[t];
                auto grad_out_l2_t = grad_out_l2[t];
                auto grad_out_l3_t = grad_out_l3[t];
                auto grad_out_l4_t = grad_out_l4[t];
                auto grad_in_t = grad_in[t];

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
                                                        assign_data_among_tensor(grad_in_t, grad_out_l4_t, scale_factor, feature_size, centre_x_l3 - 0.5, centre_x_l3 + 0.5, centre_y_l3 - 0.5, centre_y_l3 + 0.5);
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                assign_data_among_tensor(grad_in_t, grad_out_l3_t, scale_factor, feature_size, centre_x_l2 - 1, centre_x_l2 + 1, centre_y_l2 - 1, centre_y_l2 + 1);
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    assign_data_among_tensor(grad_in_t, grad_out_l2_t, scale_factor, feature_size, centre_x_l1 - 2, centre_x_l1 + 2, centre_y_l1 - 2, centre_y_l1 + 2);
                                }
                            }
                        }
                    }
                    else
                    {
                        //if not set, average the content

                        assign_data_among_tensor(grad_in_t, grad_out_l1_t, scale_factor, feature_size, centre_x - 4, centre_x + 4, centre_y - 4, centre_y + 4);
                    }
                }
            }
        });
    }

} // namespace ms
