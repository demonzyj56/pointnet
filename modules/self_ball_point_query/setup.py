"""Build external module."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='self_bpq_cuda',
    ext_modules=[
        CUDAExtension('self_bpq_cuda', [
            'self_bpq_cuda.cpp',
            'self_bpq_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
