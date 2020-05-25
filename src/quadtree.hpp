/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:46:45
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-25 23:08:07
 * @ Description:
 */
#ifndef QUADTREE
#define QUADTREE

#include <algorithm>
#include <bitset>
#include <vector>
#include <assert.h>
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

        qt_size_t num_blocks() const;

        void resize(qt_size_t _n, qt_size_t _grid_height, qt_size_t _grid_width,
                    qt_size_t _feature_size, qt_size_t _n_leafs);

        ~quadtree()
        {
            delete[] trees;
            delete[] prefix_leafs;
            delete[] data; //?
        };

    public:
        qt_size_t grid_height;  // number of quadtree grids in the height dimension.
        qt_size_t grid_width;   // number of quadtrees grids the width dimension.
        qt_size_t feature_size; // length of the data vector associated with a single cell.

        qt_size_t n_leafs; // number of leaf nodes in the complete struct.

        qt_tree_t *trees;        // array of grids x bitset<21>, each bitset encode the
                                 // structure of the quadtree grid as bit strings.
        qt_size_t *prefix_leafs; // prefix sum of the number of leafs in each quadtree grid.
        qt_data_t *data;         // data array, all feature vectors associated with
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

    inline qt_size_t quadtree::num_blocks() const
    {
        return grid_height * grid_width;
    }

    inline quadtree::~quadtree()
    {
        delete[] trees;
        delete[] prefix_leafs;
    }

    //get data index using tree bit index of particular leaf
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

    inline int child_idx(const int &bit_idx) { return 4 * bit_idx + 1; }

    inline bool tree_isset_bit(const qt_tree_t &tree, const int &idx)
    {
        return tree.test(idx);
    }

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