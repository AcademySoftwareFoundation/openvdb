# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME cutlass
    GITHUB_REPOSITORY NVIDIA/cutlass
    PATCH_COMMAND git apply --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_SOURCE_DIR}/../env/cutlass.patch || exit 0
    GIT_TAG v3.4.0
    DOWNLOAD_ONLY YES
)

# We don't build and install the cutlass package, it is header only. If we install it,
# It wants to build a bunch of tests and examples that take a long time to compile.
# Instead, we just add the headers to the include path and create an interface target.
if(cutlass_ADDED)
    add_library(cutlass INTERFACE)
    target_include_directories(cutlass INTERFACE ${cutlass_SOURCE_DIR}/include)
    target_include_directories(cutlass INTERFACE ${cutlass_SOURCE_DIR}/tools/util/include)
endif()
