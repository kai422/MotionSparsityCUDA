/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:47:48
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-18 11:52:44
 * @ Description: create quadtree structure from input HEVC dense image.
 */
#ifndef CREATION
#define CREATION

#include "commdef.hpp"
#include "quadtree.hpp"

namespace ms
{
    class CreateFromDense
    {
    public:
        CreateFromDense(qt_size_t h, qt_size_t d, qt_size_t f, const qt_data_t *data_ptr) : grid_height(h), grid_width(d), feature_size(f), data(data_ptr){};
        ~CreateFromDense();
        quadtree &operator()(bool fit = false, int fit_multiply = 1, bool pack = false, int n_threads = 1);

    private:
        quadtree &create_octree(bool fit, int fit_multiply, bool pack, int n_threads);
        const qt_size_t grid_height;
        const qt_size_t grid_width;
        const qt_size_t feature_size;
        const qt_data_t *data;
    };
} // namespace ms

#endif