/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-24 16:58:43
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-24 22:31:04
 * @ Description:
 */

#ifndef DENSETOQUAD
#define DENSETOQUAD

#include "quadtree.hpp"

namespace ms
{
    // template <typename Dtype>
    quadtree *DenseToQuad(int f, int tensor_h, int tensor_w, float *data_ptr, quadtree &stru)
    {
        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index

        // grid size 64x64
        // each grid at most 8x8 leaves(at current mv resolution)
        // each leaf has 8x8 pixels
        assert(f == stru.feature_size && ((float)tensor_h / stru.grid_height) == ((float)stru.grid_width / tensor_w) &&
               "expect input structure has same size with data tensor.");
        float scale_factor = (float)tensor_h / stru.grid_height;

        quadtree *output = new quadtree(stru);
        output->data = new qt_data_t[output->n_leafs * output->feature_size]{};

        int n_blocks = output->grid_height * output->grid_width;
        int grid_width = output->grid_width;
        int grid_height = output->grid_height;
        int feature_size = output->feature_size;
        //#pragma omp parallel for
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            bitset<21UL> &grid_tree = output->trees[grid_idx];
            qt_data_t *grid_data = output->data + output->feature_size * output->prefix_leafs[grid_idx];

            int grid_h_idx = grid_idx / grid_width;
            int grid_w_idx = grid_idx % grid_width;
            float centre_x = grid_w_idx * 8 + 4;
            float centre_y = grid_h_idx * 8 + 4;
            int data_idx = 0;

            if (tree_isset_bit(grid_tree, 0))
            {
                int bit_idx_l1 = 1;
                for (int hl1 = 0; hl1 < 2; ++hl1)
                {
                    for (int wl1 = 0; wl1 < 2; ++wl1)
                    {
                        float centre_x_l1 = centre_x + (wl1 * 4) - 2;
                        float centre_y_l1 = centre_y + (hl1 * 4) - 2;
                        if (tree_isset_bit(grid_tree, bit_idx_l1))
                        {
                            int bit_idx_l2 = child_idx(bit_idx_l1);
                            for (int hl2 = 0; hl2 < 2; ++hl2)
                            {
                                for (int wl2 = 0; wl2 < 2; ++wl2)
                                {
                                    float centre_x_l2 = centre_x_l1 + (wl2 * 2) - 1;
                                    float centre_y_l2 = centre_y_l1 + (hl2 * 2) - 1;
                                    if (tree_isset_bit(grid_tree, bit_idx_l2))
                                    {
                                        int bit_idx_l3 = child_idx(bit_idx_l2);
                                        for (int hl3 = 0; hl3 < 2; ++hl3)
                                        {
                                            for (int wl3 = 0; wl3 < 2; ++wl3)
                                            {
                                                float centre_x_l3 = centre_x_l2 + (wl3 * 1) - 0.5;
                                                float centre_y_l3 = centre_y_l2 + (hl3 * 1) - 0.5;
                                                int data_idx = tree_data_idx(grid_tree, bit_idx_l3, feature_size);
                                                get_data_from_tensor(centre_x_l3, centre_y_l3, grid_data + data_idx, data_ptr, scale_factor, tensor_h, tensor_w);
                                                bit_idx_l3++;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        int data_idx = tree_data_idx(grid_tree, bit_idx_l2, feature_size);
                                        get_data_from_tensor(centre_x_l2, centre_y_l2, grid_data + data_idx, data_ptr, scale_factor, tensor_h, tensor_w);
                                    }
                                    bit_idx_l2++;
                                }
                            }
                        }
                        else
                        {
                            int data_idx = tree_data_idx(grid_tree, bit_idx_l1, feature_size);
                            get_data_from_tensor(centre_x_l1, centre_y_l1, grid_data + data_idx, data_ptr, scale_factor, tensor_h, tensor_w);
                        }
                        bit_idx_l1++;
                    }
                }
            }
            else
            {
                get_data_from_tensor(centre_x, centre_y, grid_data + data_idx, data_ptr, scale_factor, tensor_h, tensor_w);
            }
        }

        return output;
    }

    void get_data_from_tensor(int cx, int cy, qt_data_t *dst_data_ptr, float *src_tensor, float &scale_factor, int &tensor_h, int &tensor_w)
    {
        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index
        //do pooling
        //below implementation just consider value inside grid are same so we just pick one value from it.
        //224不太好。。 256吧 如果不行再试试224
        //should implement template like avg max.
        int h_index = int(cx * scale_factor);
        int w_index = int(cy * scale_factor);
        *(dst_data_ptr) = src_tensor[0 * tensor_h * tensor_w + h_index * tensor_w + w_index];
        *(dst_data_ptr + 1) = src_tensor[1 * tensor_h * tensor_w + h_index * tensor_w + w_index];
    }
} // namespace ms

#endif