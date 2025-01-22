# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# TODO: once FVDB is built using the CMake build system, we can remove the hard-coded path
# to library and use a target name.

set(FVDB_BUILD_DIR "${CMAKE_BINARY_DIR}/../../../build/lib.linux-x86_64-cpython-310/fvdb/")
find_library(
    FVDB_LIBRARY
    NAMES fvdb
    HINTS ${FVDB_BUILD_DIR})

# check that FVDB library is found
if (FVDB_LIBRARY)
    message(STATUS "FVDB library: ${FVDB_LIBRARY}")
else()
    message(FATAL_ERROR "FVDB library not found. Please build FVDB first.")
endif()

add_library(fvdb SHARED IMPORTED)
set_target_properties(fvdb PROPERTIES IMPORTED_LOCATION ${FVDB_LIBRARY})
