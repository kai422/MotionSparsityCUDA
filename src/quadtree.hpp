/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:46:45
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-18 18:00:57
 * @ Description:
 */
#ifndef QUADTREE
#define QUADTREE

namespace ms
{
    typedef int qt_size_t;
    typedef int qt_tree_t;
    typedef float qt_data_t;
    const int N_TREE_INTS = 2;

    struct quadtree
    {
    public:
        quadtree(qt_size_t _n = 0,
                 qt_size_t _grid_height = 0,
                 qt_size_t _grid_width = 0,
                 qt_size_t _feature_size = 0,
                 qt_size_t _n_leafs = 0)
            : n(_n), grid_height(_grid_height), grid_width(_grid_width),
              feature_size(_feature_size), n_leafs(_n_leafs),
              grid_capacity(_n * _grid_height * _grid_width),
              data_capacity(_n_leafs * _feature_size)
        {
            trees = new qt_tree_t[grid_capacity * N_TREE_INTS];
            prefix_leafs = new qt_size_t[grid_capacity];
            data = new qt_data_t[data_capacity];
        };

        void resize(qt_size_t _n, qt_size_t _grid_height, qt_size_t _grid_width, qt_size_t _feature_size, qt_size_t _n_leafs);

        ~quadtree();

    private:
        qt_size_t n;            ///< number of grid-quadtrees (batch size).
        qt_size_t grid_height;  ///< number of shallow quadtrees in the height dimension.
        qt_size_t grid_width;   ///< number of shallow quadtrees in the width dimension.
        qt_size_t feature_size; ///< length of the data vector associated with a single cell.

        qt_size_t n_leafs; ///< number of leaf nodes in the complete struct.

        qt_tree_t *trees;        ///< array of length quadtree_num_blocks(grid) x N_TREE_qt_size_tS that encode the structure of the shallow quadtrees as bit strings.
        qt_size_t *prefix_leafs; ///< prefix sum of the number of leafs in each shallow quadtree.
        qt_data_t *data;         ///< contiguous data array, all feature vectors associated with the grid-quadtree data structure.

        qt_size_t grid_capacity; ///< Indicates how much memory is allocated for the trees and prefix_leafs array
        qt_size_t data_capacity; ///< Indicates how much memory is allocated for the data array
    };

    inline void quadtree::resize(qt_size_t _n, qt_size_t _grid_height, qt_size_t _grid_width, qt_size_t _feature_size, qt_size_t _n_leafs)
    {
        n = _n;
        grid_height = _grid_height;
        grid_width = _grid_width;
        feature_size = _feature_size;
        n_leafs = _n_leafs;

        qt_size_t _grid_capacity = n * grid_height * grid_width;

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
        if (data_capacity < _data_capacity)
        {
            data_capacity = _data_capacity;
            if (data != nullptr)
            {
                delete[] data;
            }
            data = new qt_data_t[data_capacity];
        }
    }

    inline quadtree::~quadtree()
    {
        delete[] trees;
        delete[] prefix_leafs;
        delete[] data;
    }
} // namespace ms

#endif