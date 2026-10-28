# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Command Line Tools

#]=======================================================================]

cmake_minimum_required(VERSION 3.24)
project(NanoVDBTests LANGUAGES CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "--------- Configuring NanoVDB Unit Test ------------")
message(STATUS "----------------------------------------------------")

###############################################################################
# -----------------------------------------------------------------------------
# workaround for win32 bug when nvcc "--keep" is used.
if(WIN32 AND NANOVDB_CUDA_KEEP_PTX)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data")

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
  # /Zc:preprocessor fixes aligned_alloc issues with CCCL on MSVC
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/bigobj /Zc:preprocessor\"")
endif()

# -----------------------------------------------------------------------------
# Compiles each installed C++ header, except the deprecated forwarding headers,
# as the first include of its own CUDA translation unit, so a header that does
# not compile on its own under nvcc (a missing or cyclic include) fails the
# build. The C headers are skipped, and tools/NanoToOpenVDB.h needs OpenVDB.

if(NANOVDB_USE_CUDA)
  set(NANOVDB_HEADER_CHECK_SKIP CNanoVDB.h PNanoVDB.h math/CSampleFromVoxels.h)
  if(NOT NANOVDB_USE_OPENVDB)
    list(APPEND NANOVDB_HEADER_CHECK_SKIP tools/NanoToOpenVDB.h)
  endif()
  set(NANOVDB_HEADER_CHECK_SOURCES)
  foreach(header ${NANOVDB_INCLUDE_FILES} ${NANOVDB_INCLUDE_IO_FILES} ${NANOVDB_INCLUDE_MATH_FILES}
                 ${NANOVDB_INCLUDE_TOOLS_FILES} ${NANOVDB_INCLUDE_UTIL_FILES} ${NANOVDB_INCLUDE_CUDA_FILES}
                 ${NANOVDB_INCLUDE_TOOLS_CUDA_FILES} ${NANOVDB_INCLUDE_UTIL_CUDA_FILES})
    if(header IN_LIST NANOVDB_HEADER_CHECK_SKIP)
      continue()
    endif()
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../${header}" deprecated REGEX "NANOVDB_DEPRECATED_HEADER")
    if(deprecated)
      continue()
    endif()
    string(MAKE_C_IDENTIFIER "${header}" name)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/header_check/${name}.cu")
    file(CONFIGURE OUTPUT "${source}" CONTENT "#include <nanovdb/${header}>\n")
    list(APPEND NANOVDB_HEADER_CHECK_SOURCES "${source}")
  endforeach()
  add_library(nanovdb_cuda_header_check OBJECT ${NANOVDB_HEADER_CHECK_SOURCES})
  target_link_libraries(nanovdb_cuda_header_check PRIVATE nanovdb)
endif()

# TODO: Benchmark should probably not require gtest.
if(NOT TARGET GTest::GTest)
  message(WARNING " - GTest required to build unittests. Skipping.")
  return()
endif()

# -----------------------------------------------------------------------------

add_executable(nanovdb_test_nanovdb "TestNanoVDB.cc")
target_link_libraries(nanovdb_test_nanovdb PRIVATE nanovdb GTest::GTest GTest::Main)
add_test(nanovdb_unit_test nanovdb_test_nanovdb)

# -----------------------------------------------------------------------------

if(NANOVDB_USE_CUDA)
  add_executable(nanovdb_test_cuda "TestNanoVDB.cu")
  target_link_libraries(nanovdb_test_cuda PRIVATE nanovdb GTest::GTest GTest::Main)
  set_target_properties(nanovdb_test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(nanovdb_cuda_unit_test nanovdb_test_cuda)

  add_executable(nanovdb_test_mgpu "TestMultiGPU.cu")
  target_link_libraries(nanovdb_test_mgpu PRIVATE nanovdb GTest::GTest GTest::Main)
  set_target_properties(nanovdb_test_mgpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(nanovdb_mgpu_unit_test nanovdb_test_mgpu)

  add_executable(nanovdb_test_cuda_memory_resource "TestMemoryResource.cu")
  target_link_libraries(nanovdb_test_cuda_memory_resource PRIVATE nanovdb GTest::GTest GTest::Main)
  set_target_properties(nanovdb_test_cuda_memory_resource PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(nanovdb_cuda_memory_resource_unit_test nanovdb_test_cuda_memory_resource)

  add_executable(nanovdb_test_cuda_util "TestUtilCuda.cu")
  target_link_libraries(nanovdb_test_cuda_util PRIVATE nanovdb GTest::GTest GTest::Main)
  set_target_properties(nanovdb_test_cuda_util PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(nanovdb_cuda_util_unit_test nanovdb_test_cuda_util)

  # Same tests with every allocation forced through cudaMalloc/cudaFree --
  # the configuration a pool-less vGPU relies on.
  add_executable(nanovdb_test_cuda_util_sync "TestUtilCuda.cu")
  target_compile_definitions(nanovdb_test_cuda_util_sync PRIVATE NANOVDB_USE_SYNC_CUDA_MALLOC)
  target_link_libraries(nanovdb_test_cuda_util_sync PRIVATE nanovdb GTest::GTest GTest::Main)
  set_target_properties(nanovdb_test_cuda_util_sync PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(nanovdb_cuda_util_sync_unit_test nanovdb_test_cuda_util_sync)
  add_executable(nanovdb_test_cuda_buffer "TestBuffer.cu")
  target_link_libraries(nanovdb_test_cuda_buffer PRIVATE nanovdb GTest::GTest GTest::Main)
  set_target_properties(nanovdb_test_cuda_buffer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(nanovdb_cuda_buffer_unit_test nanovdb_test_cuda_buffer)
endif()

# -----------------------------------------------------------------------------

if(NOT (NANOVDB_USE_TBB AND NANOVDB_USE_OPENVDB))
  message(WARNING " - OpenVDB required to build OpenVDB unit tests. Skipping.")
  return()
endif()

add_executable(nanovdb_test_openvdb "TestOpenVDB.cc")
target_link_libraries(nanovdb_test_openvdb PRIVATE nanovdb GTest::GTest GTest::Main)
add_test(nanovdb_openvdb_unit_test nanovdb_test_openvdb)
