# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Benchmark

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(NanoVDBBenchmark LANGUAGES CXX)

# -----------------------------------------------------------------------
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

set(BENCHMARK_SOURCES Benchmark.cpp Camera.h Image.h)

if(NANOVDB_USE_CUDA)
  list(APPEND BENCHMARK_SOURCES BenchKernels_nano.cu)

  add_executable(benchmark_nano Benchmark_nano.cpp BenchKernels_nano.cu)
  add_executable(benchmark_dense Benchmark_dense.cpp BenchKernels_dense.cu)

  target_include_directories(benchmark_nano PUBLIC ${CUDA_INCLUDE_DIRECTORY})
  target_include_directories(benchmark_dense PUBLIC ${CUDA_INCLUDE_DIRECTORY})

  target_link_libraries(benchmark_nano PRIVATE nanovdb ${CUDA_LIBRARIES})
  target_link_libraries(benchmark_dense PRIVATE nanovdb ${CUDA_LIBRARIES})
endif()

# -----------------------------------------------------------------------

add_executable(benchmark ${BENCHMARK_SOURCES})
target_include_directories(benchmark PUBLIC ${CUDA_INCLUDE_DIRECTORY})
target_link_libraries(benchmark PRIVATE nanovdb GTest::GTest GTest::Main)

install(TARGETS benchmark DESTINATION examples)

# ---------------------------------------------------------------
