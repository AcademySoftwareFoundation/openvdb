# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

if (DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_SOURCECODE_DIR $ENV{CPM_SOURCE_CACHE})
else()
  set(CPM_SOURCECODE_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Download CPM.cmake to the cache directory if it doesn't exist
if(NOT EXISTS ${CPM_SOURCECODE_DIR}/CPM.cmake)
  file(
    DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake
    ${CPM_SOURCECODE_DIR}/CPM.cmake
    EXPECTED_HASH SHA256=c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d
    TIMEOUT 600
    INACTIVITY_TIMEOUT 60
  )
endif()

# Include CPM.cmake from the cache
include(${CPM_SOURCECODE_DIR}/CPM.cmake)
