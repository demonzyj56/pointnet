"""Build external module."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='octant_query_cuda',
    ext_modules=[
        CUDAExtension('octant_query_cuda', [
            'octant_query_cuda.cpp',
            'octant_query_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

