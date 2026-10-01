import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

try:
    import torch
    import torch.version
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    cuda_version = torch.version.cuda

    assert cuda_version is not None, "Pytorch CUDA is required for this installation."

except ImportError:
    raise ValueError("Pytorch not found, please install it first.")

PACKAGE_NAME = "vipe"
SOURCE_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

coder_finder_path = f"{PACKAGE_NAME}/ext/specs.py"
code_finder_namespace = {"__file__": coder_finder_path}
with open(coder_finder_path, "r") as fh:
    exec(fh.read(), code_finder_namespace)
get_sources = code_finder_namespace["get_sources"]
get_cpp_flags = code_finder_namespace["get_cpp_flags"]
get_cuda_flags = code_finder_namespace["get_cuda_flags"]


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        self._copy_configs()

    def _copy_configs(self) -> None:
        if not SOURCE_CONFIG_DIR.is_dir():
            raise RuntimeError(f"Missing config source directory: {SOURCE_CONFIG_DIR}")

        target = Path(self.build_lib) / PACKAGE_NAME / "_configs"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            SOURCE_CONFIG_DIR,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (target / "__init__.py").write_text(
            "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            "# SPDX-License-Identifier: Apache-2.0\n\n"
            '"""Package data target for build-time generated ViPE configs."""\n',
            encoding="utf-8",
        )


# Setup CUDA_HOME for conda environment for consistency
if "CONDA_PREFIX" in os.environ:
    conda_nvcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc")
    if os.path.exists(conda_nvcc_path):
        os.environ["PYTORCH_NVCC"] = conda_nvcc_path

<<<<<<< HEAD
# Download the put Eigen 3.4 in a correct place
cpp_flags = list(get_cpp_flags())
cuda_flags = list(get_cuda_flags())

# Pin C++17. No standard is set otherwise, so nvcc follows the host compiler's
# default -- C++20 on GCC 13 -- which makes std::lerp visible. csrc/utils_ext/
# math_util.h:1059 then declares its own `float lerp(float, float, float)`,
# which the compiler reads as a redeclaration of std::lerp with a different
# return-type qualification and rejects:
#
#   math_util.h:1059:44: error: 'float lerp(float, float, float)' conflicts
#   with a previous declaration
#   /usr/include/c++/13/cmath:3642: note: previous declaration
#   'constexpr float std::lerp(float, float, float)'
#
# Only the scalar overload collides; the float2/float4 ones take ViPE's own
# types and are unaffected. Setting the standard rather than editing the header
# keeps the fix outside the sources, so it survives a submodule update.
if not any(flag.startswith("-std=") for flag in cpp_flags):
    cpp_flags.append("-std=c++17")
if not any(flag.startswith("-std=") for flag in cuda_flags):
    cuda_flags.append("-std=c++17")
if os.environ.get("USE_SYSTEM_EIGEN", "0") == "0":
    eigen_include_dir = "csrc/include/eigen3"
    eigen_url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"

    if not os.path.exists(eigen_include_dir):
        os.makedirs(eigen_include_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_tar_path = os.path.join(temp_dir, "eigen.gz")
            extracted_dir = os.path.join(temp_dir, "eigen-extracted")
            urlretrieve(eigen_url, tmp_tar_path)
            with tarfile.open(tmp_tar_path, "r:gz") as tar:
                tar.extractall(path=extracted_dir)

            shutil.move(os.path.join(extracted_dir, "eigen-3.4.0", "Eigen"), eigen_include_dir)

    # Use full path. We previously used "-isystem <path>" which works for GCC/clang
    # but MSVC cl.exe drops it ("ignoring unknown option '-isystem'") and then
    # treats the path as a stray source file, leaving the Eigen include unresolved.
    # Use include_dirs on the extension instead — distutils translates that into
    # the right per-compiler flag (/I on MSVC, -I on GCC/clang).
    additional_include_path = os.path.join(os.path.dirname(__file__), "csrc/include")
    include_dirs = [additional_include_path]
else:
    include_dirs = []
=======
cpp_flags = get_cpp_flags()
cuda_flags = get_cuda_flags()
>>>>>>> main

packages = find_packages()
setup(
    packages=packages,
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            f"{PACKAGE_NAME}_ext",
            sources=get_sources(),  # type: ignore
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cpp_flags, "nvcc": cuda_flags},  # type: ignore
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True), "build_py": build_py},
)
