/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:46:45
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-10 23:42:40
 * @ Description:
 */
// Copyright (c) 2017, The OctNet authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef QUADTREE
#define QUADTREE

#include <smmintrin.h>

#ifdef __CUDA_ARCH__
#define QUADTREE_FUNCTION __host__ __device__
#define FMIN(a, b) fminf(a, b)
#define FMAX(a, b) fmaxf(a, b)
#define IMIN(a, b) min(a, b)
#define IMAX(a, b) max(a, b)
#else
#define QUADTREE_FUNCTION
#define FMIN(a, b) fminf(a, b)
#define FMAX(a, b) fmaxf(a, b)
#define IMIN(a, b) (((a) < (b)) ? (a) : (b))
#define IMAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

typedef int qt_size_t;
typedef int qt_tree_t;
typedef float qt_data_t;
const int N_TREE_INTS = 2;
const int N_QUAD_TREE_T_BITS = 8 * sizeof(qt_tree_t);

struct quadtree
{
    qt_size_t n;
    qt_size_t grid_height; // number of quadtree grids in the height dimension.
    qt_size_t grid_width;  // number of quadtrees grids the width dimension.
    qt_size_t
        feature_size; // length of the data vector associated with a single cell.

    qt_size_t n_leafs; // number of leaf nodes in the complete struct.

    qt_tree_t *trees; // array of grids x bitset<21>, each bitset encode the
                      // structure of the quadtree grid as bit strings.
    qt_size_t *
        prefix_leafs; // prefix sum of the number of leafs in each quadtree grid.

    qt_data_t *data;

    qt_size_t grid_capacity; // indicates how much memory is allocated for the
                             // trees and prefix_leafs array};
};

/// Computes the number of shallow quadtrees in the grid.
/// Batchsize * height *width
/// @param grid
/// @return the number of shallow quadtrees in the grid.
QUADTREE_FUNCTION
inline int quadtree_num_blocks(const quadtree *grid)
{
    return grid->n * grid->grid_height * grid->grid_width;
}

/// Computes the flat index of the shallow octree given the subscript indices
/// grid->feature_size * octree_num_voxels(grid).
///
/// @param gn
/// @param gh
/// @param gw
/// @return flat index of shallow octree.
QUADTREE_FUNCTION
inline int quadtree_grid_idx(const quadtree *grid, const int gn, const int gh, const int gw)
{
    return (gn * grid->grid_height + gh) * grid->grid_width + gw;
}

/// Computes the offset in the grids trees array.
/// Therefore, it returns the array to the bit string that defines the shallow
/// quadtree structure.
///
/// @param grid
/// @param grid_idx flat index of the shallow octree.
/// @return array that encodes the tree structure as bit string.
QUADTREE_FUNCTION
inline qt_tree_t *quadtree_get_tree(const quadtree *grid, const qt_size_t grid_idx)
{
    return grid->trees + grid_idx * N_TREE_INTS;
}

/// Counts the number of bits == 0 in the tree array in the range [from, to).
///
/// @param tree
/// @param from
/// @param to
/// @return number of zeros in tree in the range [from, to).
QUADTREE_FUNCTION
inline int tree_cnt0(const qt_tree_t *tree, const int from, const int to)
{
    int cnt = 0;
    int from_, range_;
    unsigned int mask_;
    from_ = IMAX(from, 0);
    range_ = -from_ + IMIN(to, N_QUAD_TREE_T_BITS);
    mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_QUAD_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
    cnt += __popc(~tree[0] & mask_);
#else
    cnt += _mm_popcnt_u32(~tree[0] & mask_);
#endif

    from_ = IMAX(from - N_QUAD_TREE_T_BITS, 0);
    range_ = -from_ + IMIN(to - N_QUAD_TREE_T_BITS, N_QUAD_TREE_T_BITS);
    mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_QUAD_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
    cnt += __popc(~tree[1] & mask_);
#else
    cnt += _mm_popcnt_u32(~tree[1] & mask_);
#endif

    from_ = IMAX(from - 2 * N_QUAD_TREE_T_BITS, 0);
    range_ = -from_ + IMIN(to - 2 * N_QUAD_TREE_T_BITS, N_QUAD_TREE_T_BITS);
    mask_ = range_ <= 0 ? 0 : ((0xFFFFFFFF >> (N_QUAD_TREE_T_BITS - range_)) << from_);
#ifdef __CUDA_ARCH__
    cnt += __popc(~tree[2] & mask_);
#else
    cnt += _mm_popcnt_u32(~tree[2] & mask_);
#endif

    return cnt;
}

/// Sets the bit of num on pos to 1.
///
/// @param num
/// @param pos
/// @return updated num.
QUADTREE_FUNCTION
inline void tree_set_bit(qt_tree_t *num, const int pos)
{
    num[pos / N_QUAD_TREE_T_BITS] |= (1 << (pos % N_QUAD_TREE_T_BITS));
}

/// Computes the bit index of the parent for the given bit_idx.
/// Used to traverse a shallow quadtree.
/// @warning does not check the range of bit_idx, and will return an invalid
/// result if for example no parent exists (e.g. for bit_idx=0).
///
/// @param bit_idx
/// @return parent bit_idx of bit_idx
QUADTREE_FUNCTION
inline int tree_parent_bit_idx(const int bit_idx)
{
    return (bit_idx - 1) / 4;
}

/// Computes the bit index of the first child for the given bit_idx.
/// Used to traverse a shallow quadtree.
///
/// @param bit_idx
/// @return child bit_idx of bit_idx
QUADTREE_FUNCTION
inline int tree_child_bit_idx(const int bit_idx)
{
    return 4 * bit_idx + 1;
}

/// Checks if the bit on pos of num is set, or not.
///
/// @param num
/// @param pos
/// @return true, if bit is set, otherwise false
QUADTREE_FUNCTION
inline bool tree_isset_bit(const qt_tree_t *num, const int pos)
{
    return (num[pos / N_QUAD_TREE_T_BITS] & (1 << (pos % N_QUAD_TREE_T_BITS))) != 0;
}

/// Computes the bit_idx in a shallow quadtree as encoded in tree using the subscript
/// indices bh, and bw.
///
/// @param tree shallow quadtree structure, bit string.
/// @param bh
/// @param bw
/// @return bit_idx that corresponds to the subscript indices.
QUADTREE_FUNCTION
inline int tree_bit_idx(const qt_tree_t *tree, const int bh, const int bw)
{
    const int bit_idx = (1 + 4 + 16) +
                        (bw % 2 == 1) * 1 + (bw / 2 % 2 == 1) * 4 + (bw / 4 % 2 == 1) * 16 +
                        (bh % 2 == 1) * 2 + (bh / 2 % 2 == 1) * 8 + (bh / 4 % 2 == 1) * 32;
    //(bw / 4 % 2 == 1) * 16 at left 4x4 grid
    //(bh / 4 % 2 == 1) * 32 at below 4x4 grid
    //(bw / 2 % 2 == 1) * 4  at left 2x2 grid of 4x4 grid
    //(bh / 2 % 2 == 1) * 8  at below 2x2 grid of 4x4 grid
    //(bw % 2 == 1) * 1 at left 1x1 grid of 2x2 grid
    //(bh % 2 == 1) * 2 at below 1x1 grid of 2x2 grid
    if (tree_isset_bit(tree, tree_parent_bit_idx(bit_idx)))
    {
        return bit_idx;
        //third layer
    }
    else if (tree_isset_bit(tree, tree_parent_bit_idx(tree_parent_bit_idx(bit_idx))))
    {
        return tree_parent_bit_idx(bit_idx);
        //second layer
    }
    else if (tree_isset_bit(tree, 0))
    {
        return tree_parent_bit_idx(tree_parent_bit_idx(bit_idx));
        //first layer
    }
    else
    {
        return 0;
        //root
    }
}

/// Computes the level of the leaf in a shallow quadtree as encoded in tree using the subscript
/// indices bh, and bw.
///
/// @param tree shallow quadtree structure, bit string.
/// @param bh
/// @param bw
/// @return tree level that corresponds to the subscript indices.
inline int tree_level(const qt_tree_t *tree, const int bh, const int bw)
{
    const int bit_idx = (1 + 4 + 16) +
                        (bw % 2 == 1) * 1 + (bw / 2 % 2 == 1) * 4 + (bw / 4 % 2 == 1) * 16 +
                        (bh % 2 == 1) * 2 + (bh / 2 % 2 == 1) * 8 + (bh / 4 % 2 == 1) * 32;
    //(bw / 4 % 2 == 1) * 16 at left 4x4 grid
    //(bh / 4 % 2 == 1) * 32 at below 4x4 grid
    //(bw / 2 % 2 == 1) * 4  at left 2x2 grid of 4x4 grid
    //(bh / 2 % 2 == 1) * 8  at below 2x2 grid of 4x4 grid
    //(bw % 2 == 1) * 1 at left 1x1 grid of 2x2 grid
    //(bh % 2 == 1) * 2 at below 1x1 grid of 2x2 grid
    if (tree_isset_bit(tree, tree_parent_bit_idx(bit_idx)))
    {
        return 3;
        //third layer
    }
    else if (tree_isset_bit(tree, tree_parent_bit_idx(tree_parent_bit_idx(bit_idx))))
    {
        return 2;
        //second layer
    }
    else if (tree_isset_bit(tree, 0))
    {
        return 1;
        //first layer
    }
    else
    {
        return 0;
        //root
    }
}

#endif