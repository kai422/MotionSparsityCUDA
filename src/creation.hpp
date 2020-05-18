/**
 * @ Author: Kai Xu
 * @ Create Time: 2020-05-16 16:47:48
 * @ Modified by: Kai Xu
 * @ Modified time: 2020-05-18 18:35:51
 * @ Description: create quadtree structure from input HEVC dense image.
 *                This code is largely based on octnet.
 */
#ifndef CREATION
#define CREATION

#include "commdef.hpp"
#include "quadtree.hpp"

class QuadtreeCreateHelperCpu
{
public:
    QuadtreeCreateHelperCpu(ms::int grid_depth_, ms::int grid_height_, ms::int grid_width_) : grid_depth(grid_depth_), grid_height(grid_height_), grid_width(grid_width_), off_gd(0), off_gh(0), off_gw(0)
    {
    }
    virtual ~QuadtreeCreateHelperCpu() {}

    virtual void update_offsets(int off_gd_, int off_gh_, int off_gw_)
    {
        off_gd = off_gd_;
        off_gh = off_gh_;
        off_gw = off_gw_;
        // printf("updated helper, off: %d,%d,%d\n", off_gd, off_gh, off_gw);
    }

    virtual void update_grid_coords(int &gd, int &gh, int &gw)
    {
        gd += off_gd;
        gh += off_gh;
        gw += off_gw;
    }

    virtual int get_grid_idx(int gd, int gh, int gw)
    {
        int old_grid_idx = (gd * grid_height + gh) * grid_width + gw;
        return old_grid_idx;
    }

public:
    ms::int grid_depth;
    ms::int grid_height;
    ms::int grid_width;

    int off_gd;
    int off_gh;
    int off_gw;
};

namespace ms
{
    class CreateFromDense
    {
    public:
        CreateFromDense(int h, int w, int f, const qt_data_t *data_ptr) : grid_height(h), grid_width(w), feature_size(f), data(data_ptr), off_gh(0), off_gw(0)
        {
            grids = new quadtree(1, grids_height, grids_width, feature_size);
        };
        ~CreateFromDense();
        quadtree *operator()(bool fit = false, int fit_multiply = 1, bool pack = false, int n_threads = 1);

    private:
        quadtree &create_octree(bool fit, int fit_multiply, bool pack, int n_threads);

        quadtree *alloc_grid();
        void create_quadtree_structure(QuadtreeCreateHelperCpu *helper);
        void fit_quadtree(quadtree *grid, int fit_multiply, QuadtreeCreateHelperCpu *helper);
        void pack_quadtree(quadtree *grid, QuadtreeCreateHelperCpu *helper);
        void update_and_resize_quadtree(quadtree *grid);
        void fill_quadtree_data(quadtree *grid, bool packed, QuadtreeCreateHelperCpu *helper);

        bool is_occupied(float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, QuadtreeCreateHelperCpu *helper);
        void get_data(bool oc, float cx, float cy, float cz, float vd, float vh, float vw, int gd, int gh, int gw, QuadtreeCreateHelperCpu *helper, qt_data_t *dst);

        void update_offsets(int off_gh_, int off_gw_)
        {
            off_gh = off_gh_;
            off_gw = off_gw_;
            // printf("updated helper, off: %d,%d,%d\n", off_gd, off_gh, off_gw);
        }

        void update_grid_coords(int &gh, int &gw)
        {
            gh += off_gh;
            gw += off_gw;
        }

        int get_grid_idx(int gh, int gw)
        {
            int old_grid_idx = gh * grid_width + gw;
            return old_grid_idx;
        }

    public:
        const int grid_height;
        const int grid_width;
        const int feature_size;
        const qt_data_t *data;

        int off_gh;
        int off_gw;

        quadtree *grids;
    };

    void CreateFromDense::create_quadtree_structure(QuadtreeCreateHelperCpu *helper)
    {
    }
} // namespace ms

#endif