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
                            'data_create.cpp', 'data_create.cu',
                            'tensor_split.cpp', 'tensor_split_kernel.cu',
                            'pybind.cpp'
                        ])
      ],
      cmdclass={'build_ext': BuildExt})
