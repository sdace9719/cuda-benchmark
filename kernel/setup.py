from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cust_full_kernel',
    ext_modules=[
        CUDAExtension(
            name='cust_full_kernel', 
            sources=[
                'bindings_full.cpp',
                'kernel_full.cu',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)