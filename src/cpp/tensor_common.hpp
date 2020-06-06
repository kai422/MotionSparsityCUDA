/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-26 12:18:41
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-07 00:59:24
 * @ Description:
 */

#ifndef TENSORCOMMON
#define TENSORCOMMON

#include <torch/extension.h>
#include "quadtree.hpp"
namespace ms
{
    inline void save_data_to_tensor(qt_data_t *src_data, torch::Tensor dst_tensor, const float &scale_factor, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2)
    {
        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index
        //do pooling into one leaf

        int h1_tensor = int(h1 * scale_factor);
        int h2_tensor = int(h2 * scale_factor);
        int w1_tensor = int(w1 * scale_factor);
        int w2_tensor = int(w2 * scale_factor);

        for (int h = h1_tensor; h < h2_tensor; ++h)
        {
            for (int w = w1_tensor; w < w2_tensor; ++w)
            {
                for (int f = 0; f < feature_size; ++f)
                {
                    dst_tensor[f][h][w] = src_data[f];
                }
            }
        }
    }

    inline void get_data_from_tensor(qt_data_t *dst, const torch::Tensor data, const float &scale_factor, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2)
    {
        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index
        //do pooling into one leaf

        int h1_tensor = int(h1 * scale_factor);
        int h2_tensor = int(h2 * scale_factor);
        int w1_tensor = int(w1 * scale_factor);
        int w2_tensor = int(w2 * scale_factor);

        for (int f = 0; f < feature_size; ++f)
        {
            dst[f] = 0;
        }

        for (int h = h1_tensor; h < h2_tensor; ++h)
        {
            for (int w = w1_tensor; w < w2_tensor; ++w)
            {
                for (int f = 0; f < feature_size; ++f)
                {
                    float val;

                    val = data[f][h][w].cpu().data_ptr<float>()[0];
                    dst[f] += val;
                }
            }
        }

        float norm = (h2_tensor - h1_tensor) * (w2_tensor - w1_tensor);

        for (int f = 0; f < feature_size; ++f)
        {
            dst[f] /= norm;
        }
    }

    inline void assign_data_among_tensor(torch::Tensor dst_tensor, const torch::Tensor src_tensor, const float &scale_factor, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2)
    {

        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index
        //do pooling into one leaf

        int h1_tensor = int(h1 * scale_factor);
        int h2_tensor = int(h2 * scale_factor);
        int w1_tensor = int(w1 * scale_factor);
        int w2_tensor = int(w2 * scale_factor);

        for (int h = h1_tensor; h < h2_tensor; ++h)
        {
            for (int w = w1_tensor; w < w2_tensor; ++w)
            {
                for (int f = 0; f < feature_size; ++f)
                {
                    dst_tensor[f][h][w] = src_tensor[f][h][w];
                }
            }
        }
    }

    inline float getTensorCPUValue(torch::Tensor t, int pos = 0)
    {
        return t.cpu().data_ptr<float>()[pos];
    }
} // namespace ms
#endif