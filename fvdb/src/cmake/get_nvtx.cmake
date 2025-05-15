# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Include guard to prevent multiple inclusion
if(DEFINED _GET_NVTX_CMAKE_INCLUDED)
  return()
endif()
set(_GET_NVTX_CMAKE_INCLUDED TRUE)


CPMAddPackage(
    NAME nvtx3
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG v3.1.0-c-cpp
    GIT_SHALLOW TRUE
)
# The actual headers for NVTX v3 are often in a subdirectory like 'include/nvtx3' or similar
set(NVTX3_INCLUDE_DIR_FROM_SOURCE "${nvtx3_SOURCE_DIR}/include/")

if(EXISTS "${NVTX3_INCLUDE_DIR_FROM_SOURCE}/nvtx3/nvtx3.hpp")
  # Set nvtx3_dir to the directory where find_path will find nvtx3
  set(nvtx3_dir "${NVTX3_INCLUDE_DIR_FROM_SOURCE}" CACHE PATH "Path to NVTX3 include directory" FORCE)
  message(STATUS "get_nvtx.cmake: Forced nvtx3_dir to CACHE: ${nvtx3_dir} (from CPM source)")
else()
  message(WARNING "get_nvtx.cmake: nvtx3/nvtx3.hpp not found in ${NVTX3_INCLUDE_DIR_FROM_SOURCE}. NVTX3 from CPM might not be configured correctly for find_path.")
endif()
