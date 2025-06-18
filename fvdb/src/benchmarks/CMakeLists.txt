# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

add_custom_command(
  OUTPUT FVDB_BENCHMARKS
  COMMAND echo Running benchmarks
  #COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running fvdb benchmarks."
  USES_TERMINAL
)

# output directory
set(BENCHMARKS_BINARY_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/gbenchmarks>")

# This function takes in a benchmark name and benchmark source and handles setting all of the
# associated properties and linking to build the benchmark
function(ConfigureBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BENCHMARKS_BINARY_DIRECTORY}
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )

  target_link_libraries(
    ${CMAKE_BENCH_NAME}
    fvdb ${TORCH_LIBRARIES}
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
