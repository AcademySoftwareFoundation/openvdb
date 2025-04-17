# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Enable testing
enable_testing()

# Get GTest via CPM
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_google_test.cmake)
include(GoogleTest)

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
  target_include_directories(${CMAKE_TEST_NAME} PRIVATE ${TORCH_INCLUDE_DIRS} ${TORCH_SOURCE_INCLUDE_DIRS})

  # Register with CTest
  add_test(NAME ${CMAKE_TEST_NAME}
           COMMAND ${CMAKE_TEST_NAME}
           WORKING_DIRECTORY ${TEST_BINARY_DIRECTORY})

  # Set test properties
  set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES
    ENVIRONMENT "FVDB_EXTERNAL_TEST_DATA_PATH=${FVDB_TEST_DATA_DIR}"
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

# Configure an example test
ConfigureTest(ExampleTest "ExampleTest.cpp")

# Configure unit tests
ConfigureTest(GaussianComputeSparseInfoTest "GaussianComputeSparseInfoTest.cpp")
ConfigureTest(GaussianTileIntersectionTest "GaussianTileIntersectionTest.cpp")
ConfigureTest(GaussianComputeNanInfMaskTest "GaussianComputeNanInfMaskTest.cpp")
ConfigureTest(GaussianRasterizeBackwardTest "GaussianRasterizeBackwardTest.cpp")
ConfigureTest(GaussianRasterizeForwardTest "GaussianRasterizeForwardTest.cpp")
ConfigureTest(GaussianSphericalHarmonicsForwardTest "GaussianSphericalHarmonicsForwardTest.cpp")
ConfigureTest(GaussianSphericalHarmonicsBackwardTest "GaussianSphericalHarmonicsBackwardTest.cpp")
