from setuptools import setup, Extension

from torch.utils import cpp_extension


class BuildExt(cpp_extension.BuildExtension):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()


setup(name="MotionSparsity",
      version="0.0.1",
      ext_modules=[
          cpp_extension.CppExtension(name="MSBackend",
                                     sources=[
                                         'data_create.cpp', 
                                         #'tensor_split.cpp', 
                                         #'tensor_combine.cpp',
                                         #'grid_pool.cpp',
                                         'save_img.cpp', 'pybind.cpp'
                                     ])
      ],
      cmdclass={'build_ext': BuildExt})
