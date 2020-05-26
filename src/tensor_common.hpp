/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-26 12:18:41
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-26 21:17:05
 * @ Description:
 */

#ifndef TENSORCOMMON
#define TENSORCOMMON

#include "quadtree.hpp"
namespace ms
{
    void save_data_to_tensor(qt_data_t *src_data, float *dst_tensor, const float &scale_factor, const int &tensor_h, const int &tensor_w, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2)
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
                    float val;

                    dst_tensor[(f * tensor_h + h) * tensor_w + w] = src_data[f];
                }
            }
        }
    }

    void get_data_from_tensor(qt_data_t *dst, float *src_tensor, const float &scale_factor, const int &tensor_h, const int &tensor_w, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2)
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

                    val = src_tensor[(f * tensor_h + h) * tensor_w + w];
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

    void assign_data_among_tensor(float *dst_tensor, float *src_tensor, const float &scale_factor, const int &tensor_h, const int &tensor_w, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2)
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
                    int idx = (f * tensor_h + h) * tensor_w + w;
                    dst_tensor[idx] = src_tensor[idx];
                }
            }
        }
    }

} // namespace ms
#endif