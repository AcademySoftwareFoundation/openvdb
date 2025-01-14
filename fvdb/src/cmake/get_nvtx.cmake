# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
CPMAddPackage(
    NAME nvtx3
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG v3.1.0-c-cpp
    GIT_SHALLOW TRUE
)
set(nvtx3_dir ${nvtx3_SOURCE_DIR})
