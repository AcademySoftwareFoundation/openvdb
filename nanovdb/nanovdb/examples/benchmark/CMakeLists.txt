# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Benchmark

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(NanoVDBBenchmark LANGUAGES CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "---------- Configuring NanoVDB Benchmark -----------")
message(STATUS "----------------------------------------------------")

###############################################################################

if(WIN32 AND NANOVDB_CUDA_KEEP_PTX)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
endif()

# -----------------------------------------------------------------------
# TODO: Benchmark should probably not require gtest.
if(NOT TARGET GTest::GTest)
  message(WARNING " - GTest required to build benchmark. Skipping.")
  return()
endif()

# -----------------------------------------------------------------------
# many of the sample projects depend on a data directory. This allows Debug
# launching from the cmake binary working directory.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data")

set(BENCHMARK_SOURCES Benchmark.cc Camera.h Image.h)

if(NANOVDB_USE_CUDA)
  list(APPEND BENCHMARK_SOURCES BenchKernels_nano.cu)

  add_executable(benchmark_nano Benchmark_nano.cc BenchKernels_nano.cu)
  add_executable(benchmark_dense Benchmark_dense.cc BenchKernels_dense.cu)

  target_link_libraries(benchmark_nano PRIVATE nanovdb)
  target_link_libraries(benchmark_dense PRIVATE nanovdb)

  # Propagate MSVC Runtime from the OpenVDB library (if VDB_MSVC_RUNTIME_SELECTION
  # has been set then we're building and linking against OpenVDB, see the root
  # NanoVDB CMakeLists.txt)
  if(VDB_MSVC_RUNTIME_SELECTION)
    set_target_properties(benchmark_nano PROPERTIES
      MSVC_RUNTIME_LIBRARY ${VDB_MSVC_RUNTIME_SELECTION})
    set_target_properties(benchmark_dense PROPERTIES
      MSVC_RUNTIME_LIBRARY ${VDB_MSVC_RUNTIME_SELECTION})
  endif()
endif()

# -----------------------------------------------------------------------

add_executable(benchmark ${BENCHMARK_SOURCES})
target_link_libraries(benchmark PRIVATE nanovdb GTest::GTest GTest::Main)

# Propagate MSVC Runtime from the OpenVDB library (if VDB_MSVC_RUNTIME_SELECTION
# has been set then we're building and linking against OpenVDB, see the root
# NanoVDB CMakeLists.txt)
if(VDB_MSVC_RUNTIME_SELECTION)
  set_target_properties(benchmark PROPERTIES
    MSVC_RUNTIME_LIBRARY ${VDB_MSVC_RUNTIME_SELECTION})
endif()

install(TARGETS benchmark DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples)
