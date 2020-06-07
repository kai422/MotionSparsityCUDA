/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:11:08
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-07 23:51:12
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

#include "quadtree.hpp"
#include "grid_pool.hpp"
#include "common.hpp"

namespace ms
{

    void quadtree_pool2x2_stru_batch(ptr_wrapper<quadtree *> structures, const int n)
    {
        for (int i = 0; i < n; ++i)
        {
            quadtree_pool2x2_stru(structures[i]);
        }
    }

    void quadtree_pool2x2_stru(quadtree *in)
    {
        if (in->grid_height % 2 != 0 || in->grid_width % 2 != 0)
        {
            printf("[ERROR] quadtree_gridpool2x2_cpu grid dimension should be a multiply of 2\n");
            exit(-1);
        }
        if (in->grid_height / 2 == 0 || in->grid_width / 2 == 0)
        {
            printf("[ERROR] quadtree_gridpool2x2_cpu grid dimension have to be at least 2x2\n");
            exit(-1);
        }

        int n_blocks = in->num_blocks();
        int grid_width = in->grid_width;
        qt_tree_t out_tree[n_blocks]{};
        //#pragma omp parallel for
        for (int out_grid_idx = 0; out_grid_idx < n_blocks; ++out_grid_idx)
        {
            bitset<21UL> out_grid_tree = out_tree[out_grid_idx];
            int out_gh, out_gw;
            out_gh = out_grid_idx / grid_width;
            out_gw = out_grid_idx % grid_width;

            // first bit is always set, because out block consists of 8 in blocks
            out_grid_tree.set(0);

            int obit_idx_l1 = 1;
            for (int hgh = 0; hgh < 2; ++hgh)
            {
                for (int wgw = 0; wgw < 2; ++wgw)
                {
                    int in_gh = 2 * out_gh + hgh;
                    int in_gw = 2 * out_gw + wgw;
                    int in_grid_idx = in_gh * grid_width + in_gw;
                    bitset<21UL> &in_grid_tree = in->trees[in_grid_idx];

                    //check if first bit in in blocks is set
                    if (tree_isset_bit(in_grid_tree, 0))
                    {
                        out_grid_tree.set(obit_idx_l1);

                        int obit_idx_l2 = child_idx(obit_idx_l1);
                        for (int ibit_idx_l1 = 1; ibit_idx_l1 < 5; ++ibit_idx_l1)
                        //TODO: check check
                        {
                            //check if l1 bits are set in in blocks
                            if (tree_isset_bit(in_grid_tree, ibit_idx_l1))
                            {
                                out_grid_tree.set(obit_idx_l2);
                            }
                            obit_idx_l2++;
                        }
                    }
                    obit_idx_l1++;
                }
            }
        }
        in->n_leafs = 0;
        in->prefix_leafs[0] = 0;
        for (int grid_idx = 0; grid_idx < n_blocks; ++grid_idx)
        {
            in->trees[grid_idx] = out_tree[grid_idx];

            //update n_leafs
            in->n_leafs += in->trees[grid_idx].count() * 3 + 1; //leaves (node*4) - double counted nodes(n-1)

            //update prefix_leafs
            if (grid_idx >= 1)
            {
                in->prefix_leafs[grid_idx] = in->prefix_leafs[grid_idx - 1] + (in->trees[grid_idx - 1].count() * 3 + 1);
            }
        }
        //quadtree_pool2x2_data_avg(in, out);
    }

    // inline void quadtree_pool2x2_data_avg(const qt_data_t *data_in, qt_size_t feature_size, qt_data_t *data_out)
    // {
    //     for (int f = 0; f < feature_size; ++f)
    //     {
    //         qt_data_t avg = 0;
    //         for (int idx = 0; idx < 4; ++idx)
    //         {
    //             avg += data_in[idx * feature_size + f];
    //         }
    //         avg /= 4.f;
    //         data_out[f] = avg;
    //     }
    // }
} // namespace ms