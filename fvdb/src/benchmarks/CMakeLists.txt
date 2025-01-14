# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(fvdb_benchmarks LANGUAGES CXX CUDA)

if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_ENV_PATH $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment path: ${CONDA_ENV_PATH}")
else()
    message(FATAL_ERROR "Conda environment path not found. Please activate the fvdb conda environment.")
endif()

# Get dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_cpm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_google_benchmark.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_nvtx.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_torch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_fvdb.cmake)

add_custom_command(
  OUTPUT FVDB_BENCHMARKS
  COMMAND echo Running benchmarks
  #COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running fvdb benchmarks."
  USES_TERMINAL
)

# This function takes in a benchmark name and benchmark source and handles setting all of the
# associated properties and linking to build the benchmark
function(ConfigureBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/benchmarks>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  target_include_directories(${CMAKE_BENCH_NAME} PRIVATE "${FVDB_BUILD_DIR}/include/fvdb")
  target_include_directories(${CMAKE_BENCH_NAME} PRIVATE "${FVDB_BUILD_DIR}/include/nanovdb")
  target_include_directories(${CMAKE_BENCH_NAME} PRIVATE "${CONDA_ENV_PATH}/include/python3.10")
  target_link_libraries(
    ${CMAKE_BENCH_NAME}
    fvdb ${TORCH_LIBRARIES} ${Python3_LIBRARIES}
    benchmark::benchmark_main
    $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  add_custom_command(
    OUTPUT FVDB_BENCHMARKS
    COMMAND ${CMAKE_BENCH_NAME} --benchmark_out_format=json
            --benchmark_out=results/${CMAKE_BENCH_NAME}.json
    APPEND
    COMMENT "Adding ${CMAKE_BENCH_NAME}"
  )

  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/benchmarks/fvdb
    EXCLUDE_FROM_ALL
  )
endfunction()

# Configure the benchmarks
ConfigureBench(simple "simple/simple.cpp")
#ConfigureBench(gsplat_rasterize "gsplat_rasterize/gsplat_rasterize.cpp")
