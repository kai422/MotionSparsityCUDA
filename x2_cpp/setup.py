from setuptools import setup, Extension
from torch.utils import cpp_extension

cpp_module = cpp_extension.CppExtension('x2_cpp',
                                        sources=['x2.cpp'],
                                        extra_compile_args=['-fopenmp'],
                                        extra_link_args=['-lgomp']
                                        )

setup(name='x2_cpp',
      ext_modules=[cpp_module],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
