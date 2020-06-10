/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-26 12:18:41
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-06-09 22:59:04
 * @ Description:
 */

#ifndef TENSORCOMMON
#define TENSORCOMMON

#include <torch/extension.h>
#include "common.hpp"
#include "quadtree.hpp"
namespace ms
{
    inline void save_data_to_tensor_with_border_coords(qt_data_t *src_data, torch::Tensor dst_tensor, const float &scale_factor, int &feature_size, const float &h1, const float &h2, const float &w1, const float &w2, std::vector<std::tuple<int, int>> &border_coords)
    {
        //data_ptr accessor: f_index*(h*w) + h_index*w + w_index
        //do pooling into one leaf

        int h1_tensor = int(h1 * scale_factor);
        int h2_tensor = int(h2 * scale_factor);
        int w1_tensor = int(w1 * scale_factor);
        int w2_tensor = int(w2 * scale_factor);
        int h_size = dst_tensor.size(1);
        int w_size = dst_tensor.size(2);

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

        if (0 < h1_tensor && h2_tensor < h_size)
        {
            if (0 < w1_tensor && w2_tensor < w_size)
            {
                for (int h = h1_tensor - 1; h <= h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w1_tensor - 1});
                    border_coords.push_back(std::tuple<int, int>{h, w2_tensor});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h1_tensor - 1, w});
                    border_coords.push_back(std::tuple<int, int>{h2_tensor, w});
                }
            }
            else if (w1_tensor == 0)
            {
                for (int h = h1_tensor - 1; h <= h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w2_tensor});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h1_tensor - 1, w});
                    border_coords.push_back(std::tuple<int, int>{h2_tensor, w});
                }
            }
            else if (w2_tensor == w_size)
            {
                for (int h = h1_tensor - 1; h <= h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w1_tensor - 1});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h1_tensor - 1, w});
                    border_coords.push_back(std::tuple<int, int>{h2_tensor, w});
                }
            }
            else
            {
                std::cout << "h1: " << h1_tensor << std::endl;
                std::cout << "h2: " << h2_tensor << std::endl;
                std::cout << "w1: " << w1_tensor << std::endl;
                std::cout << "w2: " << w2_tensor << std::endl;
                TORCH_CHECK(false, "MotionSparsityError: error inside save_data_to_tensor_with_border_coords");
            }
        }
        else if (h1_tensor == 0)
        {
            if (0 < w1_tensor && w2_tensor < w_size)
            {
                for (int h = h1_tensor; h <= h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w1_tensor - 1});
                    border_coords.push_back(std::tuple<int, int>{h, w2_tensor});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h2_tensor, w});
                }
            }
            else if (w1_tensor == 0)
            {
                for (int h = h1_tensor; h <= h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w2_tensor});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h2_tensor, w});
                }
            }
            else if (w2_tensor == w_size)
            {
                for (int h = h1_tensor; h <= h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w1_tensor - 1});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h2_tensor, w});
                }
            }
            else
            {
                std::cout << "h1: " << h1_tensor << std::endl;
                std::cout << "h2: " << h2_tensor << std::endl;
                std::cout << "w1: " << w1_tensor << std::endl;
                std::cout << "w2: " << w2_tensor << std::endl;
                TORCH_CHECK(false, "MotionSparsityError: error inside save_data_to_tensor_with_border_coords");
            }
        }
        else if (h2_tensor == h_size)
        {
            if (0 < w1_tensor && w2_tensor < w_size)
            {
                for (int h = h1_tensor - 1; h < h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w1_tensor - 1});
                    border_coords.push_back(std::tuple<int, int>{h, w2_tensor});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h1_tensor - 1, w});
                }
            }
            else if (w1_tensor == 0)
            {
                for (int h = h1_tensor - 1; h < h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w2_tensor});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h1_tensor - 1, w});
                }
            }
            else if (w2_tensor == w_size)
            {
                for (int h = h1_tensor - 1; h < h2_tensor; ++h)
                {
                    border_coords.push_back(std::tuple<int, int>{h, w1_tensor - 1});
                }
                for (int w = w1_tensor; w < w2_tensor; ++w)
                {
                    border_coords.push_back(std::tuple<int, int>{h1_tensor - 1, w});
                }
            }
            else
            {
                std::cout << "h1: " << h1_tensor << std::endl;
                std::cout << "h2: " << h2_tensor << std::endl;
                std::cout << "w1: " << w1_tensor << std::endl;
                std::cout << "w2: " << w2_tensor << std::endl;
                TORCH_CHECK(false, "MotionSparsityError: error inside save_data_to_tensor_with_border_coords");
            }
        }
        else
        {
            std::cout << "h1: " << h1_tensor << std::endl;
            std::cout << "h2: " << h2_tensor << std::endl;
            std::cout << "w1: " << w1_tensor << std::endl;
            std::cout << "w2: " << w2_tensor << std::endl;
            TORCH_CHECK(false, "MotionSparsityError: error inside save_data_to_tensor_with_border_coords");
        }
    }

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
        clock_t start, end;
        start = clock();
        for (int h = h1_tensor; h < h2_tensor; ++h)
        {
            for (int w = w1_tensor; w < w2_tensor; ++w)
            {
                for (int f = 0; f < feature_size; ++f)
                {
                    dst[f] += data[f][h][w].cpu().data_ptr<float>()[0];
                }
            }
        }
        end = clock();
        std::cout << "Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "S" << std::endl;
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
        // std::cout << omp_in_parallel() << std::endl;
        // std::cout << omp_get_num_threads() << std::endl;
        // std::cout << omp_get_thread_num() << std::endl;
        // std::cout << "------------" << std::endl;
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