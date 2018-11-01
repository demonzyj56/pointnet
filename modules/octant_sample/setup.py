"""Build external module."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='octant_sample_cuda',
    ext_modules=[
        CUDAExtension('octant_sample_cuda', [
            'octant_sample_cuda.cpp',
            'octant_sample_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

