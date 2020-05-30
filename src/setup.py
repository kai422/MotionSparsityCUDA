from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='add_cpp',
      ext_modules=[cpp_extension.CppExtension('add_cpp', ['add.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})