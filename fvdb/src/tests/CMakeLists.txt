# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(fvdb_tests LANGUAGES CXX CUDA)

if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_ENV_PATH $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment path: ${CONDA_ENV_PATH}")
else()
    message(FATAL_ERROR "Conda environment path not found. Please activate the fvdb conda environment.")
endif()

# Get dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_cpm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_google_test.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_nvtx.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_torch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_fvdb.cmake)

add_custom_command(
  OUTPUT FVDB_TESTS
  COMMAND echo Running tests
  #COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running fvdb C++ tests."
  USES_TERMINAL
)

# output directory
set(TEST_BINARY_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gtests>")

# This function takes in a unit test name and test source and handles setting all of the
# associated properties and linking to build the test binary
function(ConfigureTest CMAKE_TEST_NAME)
  add_executable(${CMAKE_TEST_NAME} ${ARGN})

  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIRECTORY}
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  target_include_directories(${CMAKE_TEST_NAME} PRIVATE "${FVDB_BUILD_DIR}/include/fvdb")
  target_include_directories(${CMAKE_TEST_NAME} PRIVATE "${FVDB_BUILD_DIR}/include/nanovdb")
  target_include_directories(${CMAKE_TEST_NAME} PRIVATE "${Python3_INCLUDE_DIRS}")
  target_link_libraries(
    ${CMAKE_TEST_NAME}
    fvdb ${TORCH_LIBRARIES} ${Python3_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
    $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  add_test(NAME ${CMAKE_TEST_NAME}
         COMMAND ${CMAKE_TEST_NAME}
         WORKING_DIRECTORY ${TEST_BINARY_DIRECTORY})
  add_custom_command(
    OUTPUT FVDB_TESTS
    COMMAND ${CMAKE_TEST_NAME}
    APPEND
    COMMENT "Adding ${CMAKE_TEST_NAME}"
  )

  install(
    TARGETS ${CMAKE_TEST_NAME}
    COMPONENT testing
    DESTINATION bin/gtests/fvdb
    EXCLUDE_FROM_ALL
  )
endfunction()

enable_testing()

# Configure an example test
ConfigureTest(ExampleTest "ExampleTest.cpp")
