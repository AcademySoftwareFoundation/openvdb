# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Command Line Tools

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
project(NanoVDBTests LANGUAGES CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "--------- Configuring NanoVDB Unit Test ------------")
message(STATUS "----------------------------------------------------")

###############################################################################
# TODO: Benchmark should probably not require gtest.
if(NOT TARGET GTest::GTest)
  message(WARNING " - GTest required to build unittests. Skipping.")
  return()
endif()

# -----------------------------------------------------------------------------
# workaround for win32 bug when nvcc "--keep" is used.
if(WIN32 AND NANOVDB_CUDA_KEEP_PTX)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data")

# -----------------------------------------------------------------------------

add_executable(nanovdb_test_nanovdb "TestNanoVDB.cc")
target_link_libraries(nanovdb_test_nanovdb PRIVATE nanovdb GTest::GTest GTest::Main)
add_test(nanovdb_unit_test nanovdb_test_nanovdb)

# -----------------------------------------------------------------------------

if(NANOVDB_USE_CUDA)
  add_executable(nanovdb_test_cuda "TestNanoVDB.cu")
  target_link_libraries(nanovdb_test_cuda PRIVATE nanovdb GTest::GTest GTest::Main)
  add_test(nanovdb_cuda_unit_test nanovdb_test_cuda)
endif()

# -----------------------------------------------------------------------------

if(NOT (NANOVDB_USE_TBB AND NANOVDB_USE_OPENVDB))
  message(WARNING " - OpenVDB required to build OpenVDB unit tests. Skipping.")
  return()
endif()

add_executable(nanovdb_test_openvdb "TestOpenVDB.cc")
target_link_libraries(nanovdb_test_openvdb PRIVATE nanovdb GTest::GTest GTest::Main)
add_test(nanovdb_openvdb_unit_test nanovdb_test_openvdb)
