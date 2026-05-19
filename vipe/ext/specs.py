# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from pathlib import Path


def get_sources() -> list[str]:
    csrc_path = Path(__file__).parent.parent.parent / "csrc"
    return [str(p) for p in csrc_path.glob("**/*") if p.suffix in [".cpp", ".cu"]]


def _additional_include_flags() -> list[str]:
    if "CONDA_PREFIX" in os.environ:
        conda_include_path = os.path.join(os.environ["CONDA_PREFIX"], "include")
        if os.path.exists(conda_include_path):
            return ["-isystem", conda_include_path]
    return []


def get_cpp_flags() -> list[str]:
    # -DUSE_CUDA: torch>=2.10's compiled_autograd.h has a Windows guard at line 1111
    # `#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))` that skips an
    # if-constexpr block which triggers C2872 'std: ambiguous symbol' when included
    # together with Eigen on MSVC (Eigen extends namespace std). Without USE_CUDA the
    # broken block is compiled and breaks the build.
    return ["-O3", "-DWITH_CUDA", "-DUSE_CUDA"] + _additional_include_flags()


def get_cuda_flags() -> list[str]:
    # --use_fast_math removed: relaxed-math intrinsics on sm_120 (RTX 5090) cause
    # access-violation crashes inside lietorch_gpu.cu SE3.inv() kernel. Suspected
    # FMA / division path producing NaN -> Eigen Map misalignment crash.
    return ["-O3", "-DWITH_CUDA", "-DUSE_CUDA"] + _additional_include_flags()
