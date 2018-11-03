"""Build external module."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cxx_args = ['-std=c++11']


# atomicAdd requires compute capabilities 6.x and later for floating inputs.
nvcc_args = [
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70',
]


setup(
    name='gather_points_cuda',
    ext_modules=[
        CUDAExtension('gather_points_cuda', [
            'gather_points_cuda.cpp',
            'gather_points_cuda_kernel.cu',
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
