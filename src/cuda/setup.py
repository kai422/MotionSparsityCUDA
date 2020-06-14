from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class BuildExt(BuildExtension):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()


setup(name="MotionSparsity",
      version="0.0.1",
      ext_modules=[
          CUDAExtension(name="MSBackend",
                        sources=[
                            'data_create.cpp', 'data_create_kernel.cu',
                            'tensor_split.cpp', 'tensor_split_kernel.cu',
                            'tensor_combine.cpp', 'tensor_combine_kernel.cu',
                            'pooling_in_grid.cpp', 'pooling_in_grid_kernel.cu',
                            'grid_pool.cpp', 'grid_pool_kernel.cu',
                            'quadtree_kernel.cu',
                            'resize.cpp',
                            'save_img.cpp', 'save_img_kernel.cu',
                            'pybind.cpp',
                        ])
      ],
      cmdclass={'build_ext': BuildExt})
