/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:46:45
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-22 21:49:40
 * @ Description:
 */
#ifndef QUADTREE
#define QUADTREE

#include <algorithm>
#include <bitset>
#include <vector>
using std::bitset;
using std::vector;

namespace ms
{
    typedef int qt_size_t;
    typedef bitset<21> qt_tree_t;
    typedef float qt_data_t;
    const int N_TREE_INTS = 2;
    const int N_QUAD_TREE_T_BITS = 8 * sizeof(qt_tree_t);

    struct quadtree
    {
    public:
        quadtree(qt_size_t _grid_height = 0, qt_size_t _grid_width = 0,
                 qt_size_t _feature_size = 0, qt_size_t _n_leafs = 0)
            : grid_height(_grid_height), grid_width(_grid_width),
              feature_size(_feature_size), n_leafs(_n_leafs),
              grid_capacity(_grid_height * _grid_width),
              data_capacity(_n_leafs * _feature_size)
        {
            trees = new qt_tree_t[grid_capacity]{};
            prefix_leafs = new qt_size_t[grid_capacity]{};
        };

        quadtree(const quadtree &in)
            : grid_height(in.grid_height), grid_width(in.grid_width),
              feature_size(in.feature_size), n_leafs(in.n_leafs),
              grid_capacity(in.grid_capacity), data_capacity(in.data_capacity),
              trees(in.trees), prefix_leafs(in.prefix_leafs){};

        void resize(qt_size_t _n, qt_size_t _grid_height, qt_size_t _grid_width,
                    qt_size_t _feature_size, qt_size_t _n_leafs);

        void clr_trees();

        int tree_child_bit_idx(const int &bit_idx) const;

        ~quadtree();

    public:
        qt_size_t grid_height; // number of quadtree grids in the height dimension.
        qt_size_t grid_width;  // number of quadtrees grids the width dimension.
        qt_size_t
            feature_size; // length of the data vector associated with a single cell.

        qt_size_t n_leafs; // number of leaf nodes in the complete struct.

        qt_tree_t *trees; // array of grids x bitset<21>, each bitset encode the
                          // structure of the quadtree grid as bit strings.
        qt_size_t
            *prefix_leafs;      // prefix sum of the number of leafs in each quadtree grid.
        vector<qt_data_t> data; // data array, all feature vectors associated with
                                // the grid-quadtree data structure.

        qt_size_t grid_capacity; // indicates how much memory is allocated for the
                                 // trees and prefix_leafs array
        qt_size_t data_capacity; // indicates how much memory is allocated for the
                                 // data array
    };

    inline void quadtree::resize(qt_size_t _n, qt_size_t _grid_height,
                                 qt_size_t _grid_width, qt_size_t _feature_size,
                                 qt_size_t _n_leafs)
    {
        grid_height = _grid_height;
        grid_width = _grid_width;
        feature_size = _feature_size;
        n_leafs = _n_leafs;

        qt_size_t _grid_capacity = grid_height * grid_width;

        if (grid_capacity < _grid_capacity)
        {
            grid_capacity = _grid_capacity;
            if (trees != nullptr)
            {
                delete[] trees;
            }
            trees = new qt_tree_t[grid_capacity * N_TREE_INTS];

            if (prefix_leafs != nullptr)
            {
                delete[] prefix_leafs;
            }
            prefix_leafs = new qt_size_t[grid_capacity];
        }
        qt_size_t _data_capacity = n_leafs * feature_size;
    }

    inline void quadtree::clr_trees()
    {
        memset(trees, 0, grid_capacity * N_TREE_INTS);
    }

    inline int quadtree::tree_child_bit_idx(const int &bit_idx) const
    {
        return 4 * bit_idx + 1;
    }

    inline quadtree::~quadtree()
    {
        delete[] trees;
        delete[] prefix_leafs;
    }

    // template <typename Dtype>
    quadtree *DenseToQuad(int f, int h, int w, float *data_ptr, quadtree &stru)
    {
        // pic size 256x256
        // grid size 64x64
        // each grid at most 8x8 leaves(at current mv resolution)
        // each leaf has 8x8 pixels
        assert(f == stru.feature_size && h / 64 == stru.grid_height &&
               w / 64 == stru.grid_width &&
               "expect input structure has same size with data tensor.");

        quadtree *out = new quadtree(stru);

        int n_blocks = out->grid_height * out->grid_width;
        int grid_width = out->grid_width;
        int grid_height = out->grid_height;
        int feature_size = out->feature_size;
        //#pragma omp parallel for
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            bitset<21UL> &grid_tree = out->trees[grid_idx];
            vector<qt_data_t> &grid_data = out->data;

            int grid_h_idx = grid_idx / grid_width;
            int grid_w_idx = grid_idx % grid_width;
            float centre_x = grid_w_idx * 8 + 4;
            float centre_y = grid_h_idx * 8 + 4;
            int data_idx = 0;

            if (grid_tree.test(0))
            {
                int bit_idx_l1 = 1;
                for (int hl1 = 0; hl1 < 2; ++hl1)
                {
                    for (int wl1 = 0; wl1 < 2; ++wl1)
                    {
                        float centre_x_l1 = centre_x + (wl1 * 4) - 2;
                        float centre_y_l1 = centre_y + (hl1 * 4) - 2;
                        if (grid_tree.test(bit_idx_l1))
                        {
                            int bit_idx_l2 = bit_idx_l1 * 4 + 1;
                            for (int hl2 = 0; hl2 < 2; ++hl2)
                            {
                                for (int wl2 = 0; wl2 < 2; ++wl2)
                                {
                                    float centre_x_l2 = centre_x_l1 + (wl2 * 2) - 1;
                                    float centre_y_l2 = centre_y_l1 + (hl2 * 2) - 1;
                                    if (grid_tree.test(bit_idx_l2))
                                    {
                                        int bit_idx_l3 = bit_idx_l2 * 4 + 1;
                                        for (int hl3 = 0; hl3 < 2; ++hl3)
                                        {
                                            for (int wl3 = 0; wl3 < 2; ++wl3)
                                            {
                                                float centre_x_l3 = centre_x_l2 + (wl3 * 1) - 0.5;
                                                float centre_y_l3 = centre_y_l2 + (hl3 * 1) - 0.5;
                                                int data_idx = tree_data_idx(grid_tree, bit_idx_l3, feature_size);
                                                //get_data(oc, cx_l3, cy_l3, cz_l3, 1, 1, 1, gd, gh, gw,
                                                //         helper, data + data_idx);
                                                bit_idx_l3++;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        int data_idx = tree_data_idx(grid_tree, bit_idx_l2, feature_size);

                                        //get_data(oc, cx_l2, cy_l2, cz_l2, 2, 2, 2, gd, gh, gw, helper, data + data_idx);
                                    }
                                    bit_idx_l2++;
                                }
                            }
                        }
                        else
                        {
                            int data_idx = tree_data_idx(grid_tree, bit_idx_l1, feature_size);

                            //get_data(oc, cx_l1, cy_l1, cz_l1, 4, 4, 4, gd, gh, gw, helper,
                            //        data + data_idx);
                        }
                        bit_idx_l1++;
                    }
                }
            }
            else
            {
                //get_data(oc, cx, cy, cz, 8, 8, 8, gd, gh, gw, helper, data);
            }
        }
    }

    inline int tree_data_idx(const qt_tree_t &tree, const int bit_idx,
                             qt_size_t feature_size)
    {
        if (bit_idx == 0)
        {
            return 0;
        }
        int data_idx = bitset_count0(tree, 0, std::min(bit_idx, 21));
        // count number of zero from 0 to min(bit_idx, 21)
        if (parent_idx(bit_idx) > 1)
        {
            data_idx -= 4 * bitset_count0(tree, 1, parent_idx(bit_idx));
        }
        if (bit_idx > 20)
        {
            data_idx += bit_idx - 21;
        }
        return data_idx * feature_size;
    }

    inline int parent_idx(const int &bit_idx) { return (bit_idx - 1) / 4; }

    inline int bitset_count0(const qt_tree_t &tree, const int from, const int to)
    {
        assert(from >= 0 && to <= tree.size());
        int count = 0;
        for (int i = from; i < to; ++i)
        {
            count += !tree[i];
        }
        return count;
    }

} // namespace ms

#endif