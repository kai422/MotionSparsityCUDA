/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-26 11:40:38
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-01 22:02:15
 * @ Description:
 */

// Copyright (c) 2017, The OctNet authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef QUADTODENSE
#define QUADTODENSE

#include "quadtree.hpp"
#include "tensor_common.hpp"

namespace ms
{
    // template <typename Dtype>
    void QuadToDense(quadtree *input_quad, const int &f, const int &tensor_h, const int &tensor_w, float *out_data_dst)
    {
        int n_blocks = input_quad->num_blocks();
        int grid_height = input_quad->grid_height;
        int grid_width = input_quad->grid_width;
        int feature_size = input_quad->feature_size;

        assert(f == feature_size && ((float)tensor_h / input_quad->grid_height) == ((float)input_quad->grid_width / tensor_w) &&
               "expect input structure has same size with data tensor.");
        float scale_factor = (float)tensor_h / grid_height;
        //#pragma omp parallel for
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
                                                get_data_from_tensor(grid_data + data_idx, out_data_dst, scale_factor, tensor_h, tensor_w, feature_size, centre_x_l3 - 0.5, centre_x_l3 + 0.5, centre_y_l3 - 0.5, centre_y_l3 + 0.5);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        int data_idx = tree_data_idx(grid_tree, bit_idx_l2, feature_size);
                                        save_data_to_tensor(grid_data + data_idx, out_data_dst, scale_factor, tensor_h, tensor_w, feature_size, centre_x_l2 - 1, centre_x_l2 + 1, centre_y_l2 - 1, centre_y_l2 + 1);
                                    }
                                }
                            }
                        }
                        else
                        {
                            int data_idx = tree_data_idx(grid_tree, bit_idx_l1, feature_size);
                            save_data_to_tensor(grid_data + data_idx, out_data_dst, scale_factor, tensor_h, tensor_w, feature_size, centre_x_l1 - 2, centre_x_l1 + 2, centre_y_l1 - 2, centre_y_l1 + 2);
                        }
                    }
                }
            }
            else
            {
                //ouput whole grid(cx-4,cx+4,cy-4,cy+4) to out_l1_dst tensor
                save_data_to_tensor(grid_data, out_data_dst, scale_factor, tensor_h, tensor_w, feature_size, centre_x - 4, centre_x + 4, centre_y - 4, centre_y + 4);
            }
        }
    }
} // namespace ms

#endif