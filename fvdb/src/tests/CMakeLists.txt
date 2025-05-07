# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Enable testing
enable_testing()

# Get GTest via CPM
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_google_test.cmake)
include(GoogleTest)

# Find libpng
find_package(PNG REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/get_test_data.cmake)

# --- Define Test Utilities Library ---
add_library(fvdb_test_utils STATIC
  utils/ImageUtils.cpp
)

set_target_properties(fvdb_test_utils
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Include PRIVATE for compiling the library, PUBLIC for consumers (test executables)
# Add src dir PRIVATE to find <tests/utils/ImageUtils.h> from ImageUtils.cpp
target_include_directories(fvdb_test_utils
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src # To find <tests/utils/ImageUtils.h>
    ${PNG_INCLUDE_DIRS}     # Include directory for png.h
  # PUBLIC includes could be added if ImageUtils.h included e.g., Torch headers
  # PUBLIC
    # ${TORCH_INCLUDE_DIRS}
)

# Link dependencies PRIVATE as they are implementation details
target_link_libraries(fvdb_test_utils PRIVATE
  PNG::PNG
  ${TORCH_LIBRARIES} # Link utils against Torch
)
# --- End Test Utilities Library Definition ---

# output directory
set(TEST_BINARY_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/gtests>")

# This function takes in a unit test name and test source and handles setting all of the
# associated properties and linking to build the test binary
function(ConfigureTest CMAKE_TEST_NAME)
  # Create object library for the test
  # This allows us to separate the compilation dependencies from the link dependencies
  # So that we don't recompile all the test source files when  we only need to relink
  add_library(${CMAKE_TEST_NAME}_obj OBJECT ${ARGN})

  set_target_properties(
    ${CMAKE_TEST_NAME}_obj
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )

  # Include directories needed to COMPILE the test object library
  # Test sources need to find <tests/utils/ImageUtils.h> etc.
  target_include_directories(${CMAKE_TEST_NAME}_obj PRIVATE
    ${CMAKE_SOURCE_DIR}/src # Include src for finding <tests/utils/...>
    $<TARGET_PROPERTY:fvdb,INTERFACE_INCLUDE_DIRECTORIES> # Get includes exposed by fvdb
    # $<TARGET_PROPERTY:fvdb_test_utils,INTERFACE_INCLUDE_DIRECTORIES> # No longer needed with PRIVATE includes
    $<TARGET_PROPERTY:GTest::gtest,INTERFACE_INCLUDE_DIRECTORIES> # Get includes exposed by GTest::gtest
    $<TARGET_PROPERTY:GTest::gtest_main,INTERFACE_INCLUDE_DIRECTORIES> # Get includes exposed by GTest::gtest_main
  )

  target_compile_options(${CMAKE_TEST_NAME}_obj PRIVATE
    $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-G -Xcompiler=-O0>
    $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CXX>>:-O0>)

  target_compile_definitions(${CMAKE_TEST_NAME}_obj PRIVATE
    "FVDB_EXTERNAL_TEST_DATA_PATH=\"${FVDB_TEST_DATA_DIR}\"")

  # Create the actual test executable using the object files
  add_executable(${CMAKE_TEST_NAME} $<TARGET_OBJECTS:${CMAKE_TEST_NAME}_obj>)

  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_BINARY_DIRECTORY}
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )

  # Link the FINAL executable against necessary libraries
  target_link_libraries(
    ${CMAKE_TEST_NAME}
    fvdb
    fvdb_test_utils # Link test executable against test utils library
    # Dependencies like Torch and PNG are now linked transitively via fvdb_test_utils
    GTest::gtest
    GTest::gtest_main
    $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  # Register with CTest
  add_test(NAME ${CMAKE_TEST_NAME}
           COMMAND ${CMAKE_TEST_NAME}
           WORKING_DIRECTORY ${TEST_BINARY_DIRECTORY})

  # Set test properties
  set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES
    ENVIRONMENT "FVDB_EXTERNAL_TEST_DATA_PATH=${FVDB_TEST_DATA_DIR}"
  )

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
# Removed utils/ImageUtils.cpp from this call
ConfigureTest(GaussianRasterizeForwardTest "GaussianRasterizeForwardTest.cpp")
ConfigureTest(GaussianSphericalHarmonicsForwardTest "GaussianSphericalHarmonicsForwardTest.cpp")
ConfigureTest(GaussianSphericalHarmonicsBackwardTest "GaussianSphericalHarmonicsBackwardTest.cpp")
ConfigureTest(GaussianProjectionForwardTest "GaussianProjectionForwardTest.cpp")
ConfigureTest(GaussianProjectionBackwardTest "GaussianProjectionBackwardTest.cpp")
