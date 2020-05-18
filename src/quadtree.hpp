/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:46:45
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-18 11:22:11
 * @ Description:
 */
#ifndef QUADTREE
#define QUADTREE

namespace ms
{
    typedef int qt_size_t;
    typedef int qt_tree_t;
    typedef float qt_data_t;
    struct quadtree
    {
    public:
        qt_size_t n;            ///< number of grid-quadtrees (batch size).
        qt_size_t grid_height;  ///< number of shallow quadtrees in the height dimension.
        qt_size_t grid_width;   ///< number of shallow quadtrees in the width dimension.
        qt_size_t feature_size; ///< length of the data vector associated with a single cell.

        qt_size_t n_leafs; ///< number of leaf nodes in the complete struct.

        qt_tree_t *trees;        ///< array of length quadtree_num_blocks(grid) x N_TREE_INTS that encode the structure of the shallow quadtrees as bit strings.
        qt_size_t *prefix_leafs; ///< prefix sum of the number of leafs in each shallow quadtree.
        qt_data_t *data;         ///< contiguous data array, all feature vectors associated with the grid-quadtree data structure.

        qt_size_t grid_capacity; ///< Indicates how much memory is allocated for the trees and prefix_leafs array
        qt_size_t data_capacity; ///< Indicates how much memory is allocated for the data array
    };

} // namespace ms

#endif