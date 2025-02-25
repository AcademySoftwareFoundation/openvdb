# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME cutlass
    GITHUB_REPOSITORY NVIDIA/cutlass
    PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/../env/cutlass.patch
    GIT_TAG v3.4.0
    DOWNLOAD_ONLY YES
)

if(cutlass_ADDED)
    add_library(cutlass INTERFACE)
    target_include_directories(cutlass INTERFACE ${cutlass_SOURCE_DIR}/include)
    target_include_directories(cutlass INTERFACE ${cutlass_SOURCE_DIR}/tools/util/include)
endif()
