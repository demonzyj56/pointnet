"""Build external module."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='bpq_cuda',
    ext_modules=[
        CUDAExtension('bpq_cuda', [
            'ball_point_query_cuda.cpp',
            'ball_point_query_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
