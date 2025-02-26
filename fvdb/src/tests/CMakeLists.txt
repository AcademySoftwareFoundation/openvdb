# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

add_custom_command(
  OUTPUT FVDB_TESTS
  COMMAND echo Running tests
  #COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running fvdb C++ tests."
  USES_TERMINAL
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_test_data.cmake)

# output directory
set(TEST_BINARY_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/gtests>")

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

  target_link_libraries(
    ${CMAKE_TEST_NAME}
    fvdb ${TORCH_LIBRARIES}
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

  target_compile_options(${CMAKE_TEST_NAME} PRIVATE
    $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-G -Xcompiler=-O0>
    $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>:-O0>)

  target_compile_definitions(${CMAKE_TEST_NAME} PRIVATE
    "FVDB_EXTERNAL_TEST_DATA_PATH=\"${FVDB_TEST_DATA_DIR}\"")

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

# Configure unit tests
ConfigureTest(GaussianComputeSparseInfoTest "GaussianComputeSparseInfoTest.cpp")
ConfigureTest(GaussianTileIntersectionTest "GaussianTileIntersectionTest.cpp")
ConfigureTest(GaussianComputeNanInfMaskTest "GaussianComputeNanInfMaskTest.cpp")
ConfigureTest(GaussianRasterizeBackwardTest "GaussianRasterizeBackwardTest.cpp")
