# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Command Line Tools

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(NanoVDBExamples LANGUAGES CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "--------- Configuring NanoVDB Unit Test ------------")
message(STATUS "----------------------------------------------------")

###############################################################################
# TODO: Benchmark should probably not require gtest.
if(NOT (NANOVDB_GTEST))
  message(WARNING " - GTest required to build unittests. Skipping.")
  return()
endif()

# -----------------------------------------------------------------------------
# workaround for win32 bug when nvcc "--keep" is used.
if(WIN32 AND NANOVDB_CUDA_KEEP_PTX)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
endif()

# -----------------------------------------------------------------------------

set(UNITTEST_NANOVDB_SOURCE_FILES "TestNanoVDB.cpp")

add_executable(testNanoVDB ${UNITTEST_NANOVDB_SOURCE_FILES})

target_compile_definitions(testNanoVDB PRIVATE "-D${NANOVDB_USE_CUDA_FLAG}"
                                               "-D${NANOVDB_USE_TBB_FLAG}"
                                               "-D${NANOVDB_USE_BLOSC_FLAG}"
                                               "-D${NANOVDB_USE_ZLIB_FLAG}")

target_include_directories(testNanoVDB PUBLIC)

target_link_libraries(testNanoVDB PRIVATE nanovdb
                                          ${NANOVDB_GTEST}
                                          ${NANOVDB_TBB} # optional
                                          ${NANOVDB_BLOSC} # optional
                                          ${NANOVDB_ZLIB} # optional
)

add_test(NanoVDB_unit_test testNanoVDB)

# -----------------------------------------------------------------------------

if(NOT (TBB_FOUND AND OpenVDB_FOUND))
  message(WARNING " - OpenVDB required to build OpenVDB unit tests. Skipping.")
else()
  set(UNITTEST_OPENVDB_SOURCE_FILES "TestOpenVDB.cpp")

  set_source_files_properties(${UNITTEST_OPENVDB_SOURCE_FILES}
                              PROPERTIES COMPILE_FLAGS "-DNOMINMAX")

  add_executable(testOpenVDB ${UNITTEST_OPENVDB_SOURCE_FILES})

  target_compile_definitions(
    testOpenVDB
    PRIVATE "-D${NANOVDB_USE_OPENVDB_FLAG}" "-D${NANOVDB_USE_CUDA_FLAG}"
            "-D${NANOVDB_USE_TBB_FLAG}" "-D${NANOVDB_USE_BLOSC_FLAG}"
            "-D${NANOVDB_USE_ZLIB_FLAG}")

  target_include_directories(testOpenVDB PUBLIC)

  target_link_libraries(
    testOpenVDB
    PRIVATE libnanovdb
            ${NANOVDB_GTEST}
            ${NANOVDB_TBB} # optional
            ${NANOVDB_BLOSC} # optional
            ${NANOVDB_ZLIB} # optional
            ${NANOVDB_OPENVDB} # optional
  )

  add_test(OpenVDB_unit_test testOpenVDB)
endif()