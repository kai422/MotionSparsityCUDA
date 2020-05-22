/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:47:48
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-22 21:51:03
 * @ Description: create quadtree structure from input HEVC dense image.
 *                This code is largely based on octnet.
 */
#ifndef CREATION
#define CREATION

#include "commdef.hpp"
#include "quadtree.hpp"

namespace ms
{
    class CreateFromData
    {
    public:
        CreateFromData(int h, int w, int f, const qt_data_t *data_ptr) : grid_height(h), grid_width(w), feature_size(f), data(data_ptr), off_gh(0), off_gw(0)
        {
            grid = new quadtree(1, grid_height, grid_width, feature_size);
        };
        ~CreateFromData();
        quadtree *operator()(bool fit = false, int fit_multiply = 1, bool pack = false, int n_threads = 1);

    private:
        quadtree *create_quadtree(bool fit, int fit_multiply, bool pack);

        qt_data_t get_data(float cx, float cy) const;

        bool isHomogeneous(int centre_x_l1, int centre_y_l1, int size_x, int size_y, int gh, int gw) const;

        //create structure
        //create fillin data.
        //check if the colors of four color is the same. if it's the same then no need to exploit this grid as a subtree. set it as leaf, return true.
        //if it's not all the same then exploit this grid as a subtree.

    public:
        const int grid_height;
        const int grid_width;
        const int feature_size;
        const qt_data_t *data;

        int off_gh;
        int off_gw;

        quadtree *grid;
    };

    quadtree *CreateFromData::create_quadtree(bool fit, int fit_multiply, bool pack)
    {
        //create quadtree structure by checking the local sparsity and homogeneity.
        int n_blocks = grid_height * grid_width;
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            qt_tree_t &grid_tree = grid->trees[grid_idx];
            auto &grid_data = grid->data;
            int grid_h_idx = grid_idx / grid_width;
            int grid_w_idx = grid_idx % grid_width;
            float centre_x = grid_w_idx * 8 + 4;
            float centre_y = grid_h_idx * 8 + 4;
            int data_idx = 0;

            if (!isHomogeneous(centre_x, centre_y, 8, 8, grid_h_idx, grid_w_idx))
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
                        if (!isHomogeneous(centre_x_l1, centre_y_l1, 4, 4, grid_h_idx, grid_w_idx))
                        {
                            grid_tree.set(bit_idx_l1);
                            //set layer 1; maximum 4 times. bit[1-4]
                            int bit_idx_l2 = bit_idx_l1 * 4 + 1;
                            for (int hl2 = 0; hl2 < 2; ++hl2)
                            {
                                for (int wl2 = 0; wl2 < 2; ++wl2)
                                {
                                    float centre_x_l2 = centre_x_l1 + (wl2 * 2) - 1;
                                    float centre_y_l2 = centre_y_l1 + (hl2 * 2) - 1;
                                    if (!isHomogeneous(centre_x_l2, centre_y_l2, 2, 2, grid_h_idx, grid_w_idx))
                                    {
                                        grid_tree.set(bit_idx_l2);
                                        //set layer 2; maximum 4*4 times. bit[5-20]

                                        //fill in data at level 3
                                        int bit_idx_l3 = bit_idx_l2 * 4 + 1;
                                        for (int hl3 = 0; hl3 < 2; ++hl3)
                                        {
                                            for (int wl3 = 0; wl3 < 2; ++wl3)
                                            {
                                                float centre_x_l3 = centre_x_l2 + (wl3 * 1) - 0.5;
                                                float centre_y_l3 = centre_y_l2 + (hl3 * 1) - 0.5;

                                                for (int f = 0; f < feature_size; ++f)
                                                {
                                                    grid_data.push_back(get_data(centre_x_l3, centre_y_l3));
                                                    data_idx++;
                                                }
                                            }
                                        }
                                    }
                                    else
                                    {
                                        for (int f = 0; f < feature_size; ++f)
                                        {
                                            grid_data.push_back(get_data(centre_x_l2, centre_y_l2));
                                            data_idx++;
                                        }
                                    }
                                    bit_idx_l2++;
                                }
                            }
                        }
                        else
                        {
                            //fill in data w.r.t. the feature size.
                            for (int f = 0; f < feature_size; ++f)
                            {
                                grid_data.push_back(get_data(centre_x_l1, centre_y_l1));
                                data_idx++;
                            }
                        }
                    }
                    bit_idx_l1++;
                }
            }
            else
            {
                //whole gird is homogeneous
                //fill in data w.r.t. the feature size.
                for (int f = 0; f < feature_size; ++f)
                {
                    grid_data.push_back(get_data(centre_x, centre_y));
                    data_idx++;
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

            //assert data_idx
            assert(data_idx == (grid->trees[grid_idx].count() * 3 + 1) && "fillin data_idx wrong");
        }
    }

    qt_data_t CreateFromData::get_data(float cx, float cy) const
    {
        return 0;
    }

    bool CreateFromData::isHomogeneous(int centre_x_l1, int centre_y_l1, int size_x, int size_y, int gh, int gw) const
    {
        return true;
    }

} // namespace ms

#endif