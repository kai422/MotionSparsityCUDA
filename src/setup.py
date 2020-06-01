from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="MotionSparsity",
      version="0.0.1",
      ext_modules=[
          cpp_extension.CppExtension(name="MSBackend",
                                     sources=[
                                         'data_create.cpp', 'tensor_split.cpp',
                                         'tensor_combine.cpp', 'grid_pool.cpp',
                                         'pybind.cpp'
                                     ])
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})