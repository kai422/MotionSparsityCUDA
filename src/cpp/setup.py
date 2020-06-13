'''
 # @ Author: Kai Xu
 # @ Create Time: 2020-06-01 20:50:57
 # @ Modified by: Kai Xu
 # @ Modified time: 2020-06-07 22:43:21
 # @ Description:
 '''


from setuptools import setup, Extension

from torch.utils import cpp_extension


class BuildExt(cpp_extension.BuildExtension):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()


setup(name="MotionSparsityCPU",
      version="0.0.0",
      ext_modules=[
          cpp_extension.CppExtension(name="MSBackendCPU",
                                     sources=[
                                         'data_create.cpp',
                                         'tensor_split.cpp',
                                         'tensor_combine.cpp',
                                         'grid_pool.cpp',
                                         'grid_resize.cpp',
                                         'save_img.cpp', 'pybind.cpp',
                                         'densetoquad.cpp', 'quadtodense.cpp'
                                     ],
                                     extra_compile_args=['-fopenmp'],
                                     extra_link_args=['-lgomp'])
      ],
      cmdclass={'build_ext': BuildExt})
