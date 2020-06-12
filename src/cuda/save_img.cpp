/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-06-03 17:57:36
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-08 12:33:53
 * @ Description:
 */
#include <torch/extension.h>

#include "common.hpp"
#include "quadtree.hpp"
namespace ms
{
    // CUDA declarations
    void quadtree_cpy_trees_gpu_cpu_cuda(const qt_tree_t *src, qt_tree_t *dst, int num_blocks);

    inline void assign_pixel_to_tensor(const int &level, torch::TensorAccessor<float, 3, torch::DefaultPtrTraits> img, const float &scale_factor, const int tensor_h, const int tensor_w, int feature_size, const float h1, const float h2, const float w1, const float w2)
    {
        int h1_tensor = int(h1 * scale_factor);
        int h2_tensor = int(h2 * scale_factor);
        int w1_tensor = int(w1 * scale_factor);
        int w2_tensor = int(w2 * scale_factor);
        int r = 0;
        int g = 0;
        int b = 0;
        switch (level)
        {
        case 1:
            r = 204;
            g = 229;
            b = 255;
            break;
        case 2:
            r = 102;
            g = 178;
            b = 255;
            break;
        case 3:
            r = 0;
            g = 128;
            b = 255;
            break;
        case 4:
            r = 0;
            g = 51;
            b = 102;
            break;
        default:
            break;
        }

        for (int h = h1_tensor; h < h2_tensor; ++h)
        {
            for (int w = w1_tensor; w < w2_tensor; ++w)
            {
                img[h][w][0] = b;
                img[h][w][1] = g;
                img[h][w][2] = r;
            }
        }
    }

    void SaveQuadStruAsImg(ptr_wrapper<quadtree> stru_ptr, torch::Tensor img)
    {
        img = img.contiguous(); //t h w c
        auto dim = img.ndimension();

        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");

        auto T = img.size(0);
        auto h = img.size(1);
        auto w = img.size(2);

        int grid_height = stru_ptr->grid_height;
        int grid_width = stru_ptr->grid_width;
        int feature_size = stru_ptr->feature_size;
        int num_blocks = quadtree_num_blocks(stru_ptr.get());
        qt_tree_t *grid_tree_cpu = new qt_tree_t[N_TREE_INTS * num_blocks];
        quadtree_cpy_trees_gpu_cpu_cuda(stru_ptr->trees, grid_tree_cpu, num_blocks);

        for (auto t = 0; t < T; t++)
        {
            auto img_t = img[t];
            auto dst = img_t.accessor<float, 3>();

            float scale_factor = (float)h / (grid_height * 8);

            for (int grid_idx = 0; grid_idx < grid_height * grid_width; ++grid_idx)
            {
                int grid_h_idx = grid_idx / grid_width;
                int grid_w_idx = grid_idx % grid_width;
                int global_grid_idx = quadtree_grid_idx(stru_ptr.get(), t, grid_h_idx, grid_w_idx);
                qt_tree_t *grid_tree = grid_tree_cpu + global_grid_idx * N_TREE_INTS;

                //move from gpu to cpu

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
                                        int bit_idx_l2 = tree_child_bit_idx(bit_idx_l1) + hl2 * 2 + wl2;
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
                                                    assign_pixel_to_tensor(4, dst, scale_factor, h, w, feature_size, centre_x_l3 - 0.5, centre_x_l3 + 0.5, centre_y_l3 - 0.5, centre_y_l3 + 0.5);
                                                }
                                            }
                                        }
                                        else
                                        {
                                            assign_pixel_to_tensor(3, dst, scale_factor, h, w, feature_size, centre_x_l2 - 1, centre_x_l2 + 1, centre_y_l2 - 1, centre_y_l2 + 1);
                                        }
                                    }
                                }
                            }
                            else
                            {
                                assign_pixel_to_tensor(2, dst, scale_factor, h, w, feature_size, centre_x_l1 - 2, centre_x_l1 + 2, centre_y_l1 - 2, centre_y_l1 + 2);
                            }
                        }
                    }
                }
                else
                {
                    //if not set, average the content
                    assign_pixel_to_tensor(1, dst, scale_factor, h, w, feature_size, centre_x - 4, centre_x + 4, centre_y - 4, centre_y + 4);
                }
            }
        }
    }

} // namespace ms
