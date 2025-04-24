# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

CPMAddPackage(
    NAME blosc
    GITHUB_REPOSITORY Blosc/c-blosc
    PATCH_COMMAND git apply --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_SOURCE_DIR}/../env/blosc.patch || exit 0
    GIT_TAG v1.21.4
    OPTIONS
        "BUILD_SHARED OFF"
        "BUILD_TESTS OFF"
        "BUILD_FUZZERS OFF"
        "BUILD_BENCHMARKS OFF"
        "BLOSC_INSTALL ON"
        "CMAKE_POSITION_INDEPENDENT_CODE ON"
)

if(blosc_ADDED)
    add_library(blosc::blosc ALIAS blosc_static)
endif()
