/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:46:45
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-22 19:44:52
 * @ Description:
 */
#ifndef QUADTREE
#define QUADTREE

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
            trees = new qt_tree_t[grid_capacity]{};
            prefix_leafs = new qt_size_t[grid_capacity]{};
        };

        quadtree(const quadtree &in)
            : n(in.n), grid_height(in.grid_height), grid_width(in.grid_width),
              feature_size(in.feature_size), n_leafs(in.n_leafs),
              grid_capacity(in.grid_capacity),
              data_capacity(in.data_capacity),
              trees(in.trees),
              prefix_leafs(in.prefix_leafs){};

        void resize(qt_size_t _n, qt_size_t _grid_height, qt_size_t _grid_width, qt_size_t _feature_size, qt_size_t _n_leafs);

        void clr_trees();

        int tree_child_bit_idx(const int &bit_idx) const;

        ~quadtree();

    public:
        qt_size_t n;            //number of grid-quadtrees (batch size).
        qt_size_t grid_height;  //number of quadtree grids in the height dimension.
        qt_size_t grid_width;   //number of quadtrees grids the width dimension.
        qt_size_t feature_size; // length of the data vector associated with a single cell.

        qt_size_t n_leafs; //number of leaf nodes in the complete struct.

        qt_tree_t *trees;        //array of grids x bitset<21>, each bitset encode the structure of the quadtree grid as bit strings.
        qt_size_t *prefix_leafs; //prefix sum of the number of leafs in each quadtree grid.
        vector<qt_data_t> data;  //data array, all feature vectors associated with the grid-quadtree data structure.

        qt_size_t grid_capacity; //indicates how much memory is allocated for the trees and prefix_leafs array
        qt_size_t data_capacity; //indicates how much memory is allocated for the data array
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

    template <typename Dtype>
    quadtree *DenseToQuad(int bs, int ch, int h, int w, Dtype *data_ptr, quadtree &stru)
    {
        //pic size 256x256
        //grid size 64x64
        //each grid at most 8x8 leaves(at current mv resolution)
        //each leaf has 8x8 pixels
        assert(bs == stru->n && ch == stru->feature_size && h / 64 == stru->grid_height && w / 64 == stru->grid_width && "expect input structure has same size with data tensor.");
        quadtree *out = new quadtree(stru);
    }
} // namespace ms

#endif