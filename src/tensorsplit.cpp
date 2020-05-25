/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 11:46:16
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-25 10:45:41
 * @ Description: split dense tensor to three sparse tensors with hierarchy of different depths.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "quadtree.hpp"
#include "densetoquad.hpp"

namespace ms
{

    //input Tensor in
    //output Tensor out1 with only first layer.
    //output Tensor out2 with only second layer.
    //output Tensor out3 with only third layer.
    //template <typename Dtype>
    void DenseSplitForwardCPU(at::Tensor &input_r, at::Tensor &out1,
                              at::Tensor &out2, at::Tensor &out3, const quadtree &stru)
    {
        auto input = input_r;
        auto dim = input.ndimension();
        //c10::IntArrayRef input_sizes = input.sizes();
        TORCH_CHECK(dim == 4, "MotionSparsityError: expected 3D tensor, but got tensor with ", dim, " dimensions instead");
        input = input.contiguous();

        auto T = input.size(0);
        auto f = input.size(1);
        auto h = input.size(2);
        auto w = input.size(3);
        at::parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
            for (auto t = start; t < end; t++)
            {
                auto input_t = input[t];
                //create from dense
                quadtree *input_quad;
                input_quad = DenseToQuad(f, h, w, input_t.template data_ptr<float>(), stru);

                //split to three tensor

                //check quad to dense

                //split
                quadtree *out_tree_1;
                quadtree *out_tree_2;
                quadtree *out_tree_3;
                //input_quad.split(out_tree_1, out_tree_2, out_tree_3);

                //quad to dense
                //out1 = out_tree_1.toDense();
                // out2 = out_tree_2.toDense();
                //out3 = out_tree_3.toDense();
            }
        
        }
    }

    // /// Computes the subscript indices for a dense volume for the grid-octree
    // /// structure grid given the grid_idx and bit_idx.
    // ///
    // /// @param grid
    // /// @param grid_idx
    // /// @param bit_idx
    // /// @param n
    // /// @param h
    // /// @param w
    // /// @return depth of the corresponding shallow octree cell (bit_idx).
    // QUADTREE_FUNCTION
    // inline int quadtree_ind_to_dense_ind(const quadtree *grid, const int grid_idx, const int bit_idx, int *n, int *h, int *w)
    // {
    //     quadtree_split_grid_idx(grid, grid_idx, n, h, w);
    //     h[0] *= 8;
    //     w[0] *= 8;

    //     int depth = depth_from_bit_idx(bit_idx);
    //     if (depth == 1)
    //     {
    //         bhw_from_idx_l1(bit_idx, h, w);
    //     }
    //     else if (depth == 2)
    //     {
    //         bhw_from_idx_l2(bit_idx, h, w);
    //     }
    //     else if (depth == 3)
    //     {
    //         bhw_from_idx_l3(bit_idx, h, w);
    //     }

    //     return depth;
    // }
    //TODO:pooling in the grid.
    //TODO:grid pooling to adapt grid tree size.
    // template <int dense_format>
    // QUADTREE_FUNCTION inline void dense_to_quadtree_sum_fcn(const qt_data_t *dense, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int h1, int h2, int w1, int w2, qt_data_t *out)
    // {
    //     for (int f = 0; f < feature_size; ++f)
    //     {
    //         out[f] = 0;
    //     }

    //     for (int h = h1; h < h2; ++h)
    //     {
    //         for (int w = w1; w < w2; ++w)
    //         {
    //             for (int f = 0; f < feature_size; ++f)
    //             {
    //                 float val;
    //                 if (dense_format == DENSE_FORMAT_HWC)
    //                 {
    //                     val = dense[((n * dense_height + h) * dense_width + w) * feature_size + f];
    //                 }
    //                 else if (dense_format == DENSE_FORMAT_CHW)
    //                 {
    //                     val = dense[((n * feature_size + f) * dense_height + h) * dense_width + w];
    //                 }
    //                 out[f] += val;
    //             }
    //         }
    //     }
    //     //pool all h1-h2 w1-w2 data into one leaf. (cell)
    // }

    // /// Pools (avg) all the values of the tensor data dense into the corresponding
    // /// shallow quadtree cell (out).
    // ///
    // /// @tparam dense_format HWC or CHW.
    // /// @param dense the data of the tensor.
    // /// @param n batch index.
    // /// @param dense_height the height of the tensor.
    // /// @param dense_width the width of the tensor.
    // /// @param w1 start index for the pooling in width.
    // /// @param w2 end index for the pooling in width.
    // /// @param h1 end index for the pooling in height.
    // /// @param h2 end index for the pooling in height.
    // /// @param out data array of the output octree cell.
    // template <int dense_format>
    // QUADTREE_FUNCTION inline void dense_to_quadtree_avg_fcn(const qt_data_t *dense, int n, int dense_depth, int dense_height, int dense_width, int feature_size, int h1, int h2, int w1, int w2, qt_data_t *out)
    // {
    //     dense_to_quadtree_sum_fcn<dense_format>(dense, n, dense_depth, dense_height, dense_width, feature_size, h1, h2, w1, w2, out);
    //     float norm = (h2 - h1) * (w2 - w1);
    //     for (int f = 0; f < feature_size; ++f)
    //     {
    //         out[f] /= norm;
    //     }
    // }

    //template <int dense_format>
    void quadtree_to_dense_cpu(const quadtree *grid, const int dense_height, const int dense_width, qt_data_t *out_data)
    {
        int n_blocks = grid->num_blocks();
        int grid_height = grid->grid_height;
        int grid_width = grid->grid_width;

        if (dense_height < grid_height * 8 || dense_width < grid_width * 8)
        {
            printf("[ERROR] dense dim (%d,%d) is smaller then dim of grid (%d,%d,%d)\n",
                   dense_height, dense_width, grid_height * 8, grid_width * 8);
            exit(-1);
        }
        int vx_height_off = (dense_height - grid_height * 8) / 2;
        int vx_width_off = (dense_width - grid_width * 8) / 2;
        //offset from two sides
        int feature_size = grid->feature_size;

        int n_dense_elems = feature_size * dense_height * dense_width;
        //#pragma omp parallel for
        for (int idx = 0; idx < n_dense_elems; ++idx)
        {
            out_data[idx] = 0;
        }

        //#pragma omp parallel for
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            int gh = grid_idx / grid_width;
            int gw = grid_idx % grid_width;
            bitset<21UL> &tree = grid->trees[grid_idx];
            for (int bh = 0; bh < 8; ++bh)
            {
                for (int bw = 0; bw < 8; ++bw)
                {
                    int vx_h = (gh * 8) + bh + vx_height_off;
                    int vx_w = (gw * 8) + bw + vx_width_off;

                    int bit_idx = tree_bit_idx(tree, bh, bw);
                    int data_idx = tree_data_idx(tree, bit_idx, feature_size);
                    //？
                    const qt_data_t *data = grid->data + grid->feature_size * grid->prefix_leafs[grid_idx] + data_idx;

                    for (int f = 0; f < feature_size; ++f)
                    {
                        qt_data_t val = data[f];
                        //dense_format == DENSE_FORMAT_CHW)
                        int out_idx = ((feature_size + f) * dense_height + vx_h) * dense_width + vx_w;
                        out_data[out_idx] = val;
                    }
                }
            }
        }
    }

    //get the tree bit index of the dense coordinates.
    inline int tree_bit_idx(const qt_tree_t &tree, const int &bh, const int &bw)
    {
        //bh bw: the height and width offset inside the grid.
        const int bit_idx = (1 + 4 + 16) +
                            (bh / 4 % 2 == 1) * 32 + (bw / 4 % 2 == 1) * 16 +
                            (bh / 2 % 2 == 1) * 8 + (bw / 2 % 2 == 1) * 4 +
                            (bh % 2 == 1) * 2 + (bw % 2 == 1) * 1;
        //1+4+16:
        //(bh / 4 % 2 == 1) * 32 at lower grid
        //(bw / 4 % 2 == 1) * 16 at righter grid
        //(bh / 2 % 2 == 1) * 8  at lower grid of above gird
        //(bw / 2 % 2 == 1) * 4  at righter grid of above gird
        //(bh % 2 == 1) * 2 at lower grid of above gird of above above grid
        //(bw % 2 == 1) * 1 at righter grid of above gird of above above grid
        if (tree_isset_bit(tree, parent_idx(bit_idx)))
        {
            return bit_idx;
            //third layer
        }
        else if (tree_isset_bit(tree, parent_idx(parent_idx(bit_idx))))
        {
            return parent_idx(bit_idx);
            //second layer
        }
        else if (tree_isset_bit(tree, 0))
        {
            return parent_idx(parent_idx(bit_idx));
            //first layer
        }
        else
        {
            return 0;
            //root
        }
    }
} //end namespace ms.