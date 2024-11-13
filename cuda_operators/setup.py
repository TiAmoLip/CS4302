from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = "custom_matmul",
    include_dirs=['include',"headers"],
    ext_modules=[
        CUDAExtension(
            "custom_matmul",
            ["kernel/custom_matmul.cpp", "kernel/matmul.cu"],
        )
    ],
    cmdclass= {
        "build_ext": BuildExtension
    }
)