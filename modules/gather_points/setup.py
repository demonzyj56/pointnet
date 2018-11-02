"""Build external module."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='gather_points_cuda',
    ext_modules=[
        CUDAExtension('gather_points_cuda', [
            'gather_points_cuda.cpp',
            'gather_points_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
