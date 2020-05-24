/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:47:48
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-24 22:44:50
 * @ Description: create quadtree structure from input HEVC dense image.
 *                This code is largely based on octnet.
 */
#ifndef CREATION
#define CREATION

#include "quadtree.hpp"
#include <opencv2/core/core.hpp>
#include <math.h>

namespace ms
{
    quadtree *CreateFromImg(cv::Mat mv_x, cv::Mat mv_y, int ih = 256, int iw = 256, int f = 2)
    {
        //TODO:img operation

        //TODO:img operation
        int grid_h = 4;
        int grid_w = 4;
        CreateFromData create(grid_h, grid_w, f, mv_x, mv_y, ih, iw);
        return create();
    }

    class CreateFromData
    {
    public:
        CreateFromData(int gh, int gw, int f, const cv::Mat _img_x, const cv::Mat _img_y, int ih, int iw) : grid_height(gh), grid_width(gw), feature_size(f), img_ch1(_img_x), img_ch2(_img_y), img_height(ih), img_width(iw)
        {
            grid = new quadtree(grid_height, grid_width, feature_size);
            assert(img_height / (grid_height * 8) == img_width / (grid_width * 8));
            scale_factor = (float)img_height / (grid_height * 8);
        };
        ~CreateFromData();
        quadtree *operator()();

    private:
        void create_quadtree_structure();
        void fillin_data();
        void get_data(float centre_x, float centre_y, qt_data_t *dst_data_ptr) const;
        bool isHomogeneous(float centre_x, float centre_y, float size) const;

    public:
        const int img_height;
        const int img_width;
        const int grid_height;
        const int grid_width;
        const int feature_size;
        float scale_factor;
        const cv::Mat img_ch1;
        const cv::Mat img_ch2;
        quadtree *grid;
    };

    quadtree *CreateFromData::operator()()
    {
        create_quadtree_structure();
        fillin_data();
        return grid;
    }
    void CreateFromData::create_quadtree_structure()
    {
        //create quadtree structure by checking the local sparsity and homogeneity.
        int n_blocks = grid_height * grid_width;
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            qt_tree_t &grid_tree = grid->trees[grid_idx];
            int grid_h_idx = grid_idx / grid_width;
            int grid_w_idx = grid_idx % grid_width;
            float centre_x = grid_w_idx * 8 + 4;
            float centre_y = grid_h_idx * 8 + 4;

            if (!isHomogeneous(centre_x, centre_y, 8))
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
                        if (!isHomogeneous(centre_x_l1, centre_y_l1, 4))
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
                                    if (!isHomogeneous(centre_x_l2, centre_y_l2, 2))
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

            //allocate data memory area
            grid->data = new qt_data_t[grid->n_leafs * grid->feature_size]{};
        }
    }

    void CreateFromData::fillin_data()
    {
        //create quadtree structure by checking the local sparsity and homogeneity.
        int n_blocks = grid_height * grid_width;
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            qt_tree_t &grid_tree = grid->trees[grid_idx];
            qt_data_t *grid_data = grid->data + grid->feature_size * grid->prefix_leafs[grid_idx];
            int grid_h_idx = grid_idx / grid_width;
            int grid_w_idx = grid_idx % grid_width;
            float centre_x = grid_w_idx * 8 + 4;
            float centre_y = grid_h_idx * 8 + 4;

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
                                                get_data(centre_x_l3, centre_y_l3, grid_data + data_idx);
                                                bit_idx_l3++;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        int data_idx = tree_data_idx(grid_tree, bit_idx_l2, feature_size);
                                        get_data(centre_x_l2, centre_y_l2, grid_data + data_idx);
                                    }
                                    bit_idx_l2++;
                                }
                            }
                        }
                        else
                        {
                            int data_idx = tree_data_idx(grid_tree, bit_idx_l1, feature_size);
                            get_data(centre_x_l1, centre_y_l1, grid_data + data_idx);
                        }
                        bit_idx_l1++;
                    }
                }
            }
            else
            {
                //get data l0
                get_data(centre_x, centre_y, grid_data);
            }
        }
    }

    void CreateFromData::get_data(float centre_x, float centre_y, qt_data_t *dst_data_ptr) const
    {
        *(dst_data_ptr) = img_ch1.at<uchar>(int(centre_x * scale_factor), int(centre_y * scale_factor));
        *(dst_data_ptr + 1) = img_ch2.at<uchar>(int(centre_x * scale_factor), int(centre_y * scale_factor));
    }

    bool CreateFromData::isHomogeneous(float centre_x, float centre_y, float size) const
    {
        //is the value inside this grid is all the same.
        //check if the colors of four corner is the same. if it's the same then no need to exploit this grid as a subtree. set it as leaf, return true.
        //if it's not all the same then exploit this grid as a subtree.
        float x_broder_left = centre_x - size / 2;
        float x_broder_right = centre_x + size / 2;
        float y_broder_up = centre_y - size / 2;
        float y_broder_down = centre_y + size / 2;

        int x_start = ceil((x_broder_left + 0.5) * scale_factor);
        int x_end = floor((x_broder_right - 0.5) * scale_factor);
        int y_start = ceil((y_broder_up + 0.5) * scale_factor);
        int y_end = floor((y_broder_down - 0.5) * scale_factor);

        bool isHomo = true;
        /*
            x_start,y_start -------------------- x_start, y_end
            \                                               \
            \                                               \
            \                                               \
            \                                               \
            x_end,y_start  ---------------------- x_end, y_end
        */
        bool isHomo_ch1 = (img_ch1.at<uchar>(x_start, y_start) == img_ch1.at<uchar>(x_start, y_end)) && (img_ch1.at<uchar>(x_end, y_start) == img_ch1.at<uchar>(x_end, y_end)) && (img_ch1.at<uchar>(x_start, y_start) == img_ch1.at<uchar>(x_end, y_start));

        bool isHomo_ch2 = (img_ch2.at<uchar>(x_start, y_start) == img_ch2.at<uchar>(x_start, y_end)) && (img_ch2.at<uchar>(x_end, y_start) == img_ch2.at<uchar>(x_end, y_end)) && (img_ch2.at<uchar>(x_start, y_start) == img_ch2.at<uchar>(x_end, y_start));

        return isHomo_ch1 & isHomo_ch2;
    }

} // namespace ms

#endif