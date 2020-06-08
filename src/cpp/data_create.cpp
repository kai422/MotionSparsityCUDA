/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:47:48
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-08 23:30:58
 * @ Description: create quadtree structure from input HEVC dense image.
 *                This code is largely based on octnet.
 *                based on octnet.
 */

#include "quadtree.hpp"
#include "data_create.hpp"
#include "common.hpp"
#include <torch/extension.h>
#include <math.h>

namespace ms
{

    ptr_wrapper<quadtree *> CreateFromDense(torch::Tensor input)
    {
        auto dim = input.ndimension();
        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        auto T = input.size(0);
        auto f = input.size(1);
        auto h = input.size(2);
        auto w = input.size(3);
        TORCH_CHECK(f == 2, "MotionSparsityError: expected 2 channel tensor");
        TORCH_CHECK(h == 256, "MotionSparsityError: expected tensor with height 256");
        TORCH_CHECK(w == 256, "MotionSparsityError: expected tensor with width 256");
        int grid_h = 4;
        int grid_w = 4;
        quadtree **ptr_strus = new quadtree *[T] {};
        //int64_t start = 0;
        //int64_t end = T;
        parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
            for (auto t = start; t < end; t++)
            {

                auto input_t = input[t];
                quadtree **ptr_stru_t = ptr_strus + t;
                create_quadtree_structure(grid_h, grid_w, f, input_t, h, w, ptr_stru_t);
                //convert gridtree to dense tensor
            }
        });

        return ptr_strus;
    }

    void create_quadtree_structure(const int &grid_h, const int &grid_w, const int &f, const torch::Tensor data, const int &tensor_h, const int &tensor_w, quadtree **ptr_stru_t)
    {
        quadtree *grid = new quadtree(grid_h, grid_w, f);
        *ptr_stru_t = grid;
        float scale_factor = (float)tensor_h / (grid_h * 8); //should be 256/(8*4)=8

        //create quadtree structure by checking the local sparsity and homogeneity of input_tensor.
        int n_blocks = grid_h * grid_w;
#pragma omp parallel for
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            qt_tree_t &grid_tree = grid->trees[grid_idx];
            int grid_h_idx = grid_idx / grid_w;
            int grid_w_idx = grid_idx % grid_w;
            float centre_x = grid_w_idx * 8 + 4;
            float centre_y = grid_h_idx * 8 + 4;

            if (!isHomogeneous(data, centre_x, centre_y, 8, scale_factor))
            {
                grid_tree.set(0);
                //set root; bit[0]
                int bit_idx_l1 = 1;
                for (int hl1 = 0; hl1 < 2; ++hl1)
                {
                    for (int wl1 = 0; wl1 < 2; ++wl1)
                    {
                        float centre_x_l1 = centre_x + (wl1 * 4) - 2;
                        float centre_y_l1 = centre_y + (hl1 * 4) - 2;
                        if (!isHomogeneous(data, centre_x_l1, centre_y_l1, 4, scale_factor))
                        {
                            grid_tree.set(bit_idx_l1);
                            //set layer 1; maximum 4 times. bit[1-4]
                            int bit_idx_l2 = child_idx(bit_idx_l1);
                            for (int hl2 = 0; hl2 < 2; ++hl2)
                            {
                                for (int wl2 = 0; wl2 < 2; ++wl2)
                                {
                                    float centre_x_l2 = centre_x_l1 + (wl2 * 2) - 1;
                                    float centre_y_l2 = centre_y_l1 + (hl2 * 2) - 1;
                                    if (!isHomogeneous(data, centre_x_l2, centre_y_l2, 2, scale_factor))
                                    {
                                        grid_tree.set(bit_idx_l2);
                                        //set layer 2; maximum 4*4 times. bit[5-20]
                                    }
                                    bit_idx_l2++;
                                }
                            }
                        }
                        bit_idx_l1++;
                    }
                }
            }
            //to encode a full depth tree in a single grid. we will need maximum 1+4+4*4 = 21 bits to store tree information. i.e. where the leaf, which node has children. bitset is a better choice.

            //update n_leafs
            grid->n_leafs += grid->trees[grid_idx].count() * 3 + 1; //leaves (node*4) - double counted nodes(n-1)

            //update prefix_leafs
            if (grid_idx >= 1)
            {
                grid->prefix_leafs[grid_idx] = grid->prefix_leafs[grid_idx - 1] + (grid->trees[grid_idx - 1].count() * 3 + 1);
            }
        }
    }

    bool isHomogeneous(const torch::Tensor data, const float &centre_x, const float &centre_y, const float &size, const float &scale_factor)
    {
        //is the value inside this grid is all the same.
        //check if the colors of four corner is the same. if it's the same then no need to exploit this grid as a subtree. set it as leaf, return true.
        //if it's not all the same then exploit this grid as a subtree.
        //channel, x, y.
        float x_broder_left = centre_x - size / 2;
        float x_broder_right = centre_x + size / 2;
        float y_broder_up = centre_y - size / 2;
        float y_broder_down = centre_y + size / 2;

        int x_start = ceil((x_broder_left + 0.5) * scale_factor);
        int x_end = floor((x_broder_right - 0.5) * scale_factor);
        int y_start = ceil((y_broder_up + 0.5) * scale_factor);
        int y_end = floor((y_broder_down - 0.5) * scale_factor);

        /*
            x_start,y_start -------------------- x_start, y_end
            \                                               \
            \                                               \
            \                                               \
            \                                               \
            x_end,y_start  ---------------------- x_end, y_end
        */

        auto ss0 = data[0][x_start][y_start].cpu().data_ptr<float>()[0];
        auto se0 = data[0][x_start][y_end].cpu().data_ptr<float>()[0];
        auto es0 = data[0][x_end][y_start].cpu().data_ptr<float>()[0];
        auto ee0 = data[0][x_end][y_end].cpu().data_ptr<float>()[0];

        bool isHomo_ch1 = (ss0 == se0) && (es0 == ee0) && (ss0 == es0);

        auto ss1 = data[1][x_start][y_start].cpu().data_ptr<float>()[0];
        auto se1 = data[1][x_start][y_end].cpu().data_ptr<float>()[0];
        auto es1 = data[1][x_end][y_start].cpu().data_ptr<float>()[0];
        auto ee1 = data[1][x_end][y_end].cpu().data_ptr<float>()[0];

        bool isHomo_ch2 = (ss1 == se1) && (es1 == ee1) && (ss1 == es1);

        return isHomo_ch1 && isHomo_ch2;
    }

} // namespace ms
