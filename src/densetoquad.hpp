/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-24 16:58:43
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-25 10:45:22
 * @ Description:
 */

#ifndef DENSETOQUAD
#define DENSETOQUAD

#include "quadtree.hpp"

namespace ms
{
    // template <typename Dtype>
    quadtree *DenseToQuad(int f, int tensor_h, int tensor_w, float *data_ptr, const quadtree &stru)
    {
        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index
        // tensor_size tensor_h x tensor_w (256x256)
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
                                                get_data_from_tensor(grid_data + data_idx, data_ptr, scale_factor, tensor_h, tensor_w, feature_size, centre_x_l2 - 0.5, centre_x_l2 + 0.5, centre_y_l2 - 0.5, centre_y_l2 + 0.5);
                                            }
                                        }
                                    }
                                    else
                                    {
                                        int data_idx = tree_data_idx(grid_tree, bit_idx_l2, feature_size);
                                        get_data_from_tensor(grid_data + data_idx, data_ptr, scale_factor, tensor_h, tensor_w, feature_size, centre_x_l2 - 1, centre_x_l2 + 1, centre_y_l2 - 1, centre_y_l2 + 1);
                                    }
                                }
                            }
                        }
                        else
                        {
                            int data_idx = tree_data_idx(grid_tree, bit_idx_l1, feature_size);
                            get_data_from_tensor(grid_data + data_idx, data_ptr, scale_factor, tensor_h, tensor_w, feature_size, centre_x_l1 - 2, centre_x_l1 + 2, centre_y_l1 - 2, centre_y_l1 + 2);
                        }
                    }
                }
            }
            else
            {
                //if not set, average the content
                get_data_from_tensor(grid_data + data_idx, data_ptr, scale_factor, tensor_h, tensor_w, feature_size, centre_x - 4, centre_x + 4, centre_y - 4, centre_y + 4);
            }
        }

        return output;
    }

    void get_data_from_tensor(qt_data_t *dst, float *src_tensor, const float &scale_factor, const int &tensor_h, const int &tensor_w, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2)
    {
        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index
        //do pooling into one leaf

        int h1_tensor = int(h1 * scale_factor);
        int h2_tensor = int(h2 * scale_factor);
        int w1_tensor = int(w1 * scale_factor);
        int w2_tensor = int(w2 * scale_factor);

        for (int f = 0; f < feature_size; ++f)
        {
            dst[f] = 0;
        }

        for (int h = h1_tensor; h < h2_tensor; ++h)
        {
            for (int w = w1_tensor; w < w2_tensor; ++w)
            {
                for (int f = 0; f < feature_size; ++f)
                {
                    float val;

                    val = src_tensor[(f * tensor_h + h) * tensor_w + w];
                    dst[f] += val;
                }
            }
        }

        float norm = (h2_tensor - h1_tensor) * (w2_tensor - w1_tensor);

        for (int f = 0; f < feature_size; ++f)
        {
            dst[f] /= norm;
        }
    }
} // namespace ms

#endif