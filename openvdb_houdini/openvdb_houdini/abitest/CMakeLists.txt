# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for the OpenVDB Houdini ABI Test

  This test verifies ABI compatibility by constructing VDB objects using
  this version of VDB and then validating using the version of VDB shipped
  with Houdini and vice-versa. This assumes that the ABI between the two
  is the same.

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
project(OpenVDBHoudiniABITest)


#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "-------- Configuring OpenVDBHoudiniABITest ---------")
message(STATUS "----------------------------------------------------")

##########################################################################

# This binary is composed of two object files:
# - TestABI uses headers from this version of OpenVDB
# - main uses headers from the version of OpenVDB shipped with Houdini

if(NOT DEFINED HOUDINI_INCLUDE_DIR OR NOT DEFINED HOUDINI_LIB_DIR)
  message(FATAL_ERROR "The Houdini ABI Test requires the Houdini include "
    "directory and lib directory variables to be defined"
  )
endif()

# Find libopenvdb_sesi library
find_library(OPENVDB_SESI_LIB openvdb_sesi
  ${_FIND_HOUDINI_ADDITIONAL_OPTIONS}
  PATHS ${HOUDINI_LIB_DIR}
)

# Find sesi half library (needed <= H19.0)
find_library(SESI_HALF Half
  ${_FIND_HOUDINI_ADDITIONAL_OPTIONS}
  PATHS ${HOUDINI_LIB_DIR}
)

if(NOT EXISTS ${OPENVDB_SESI_LIB})
  message(FATAL_ERROR "The Houdini ABI Test is unable to locate libopenvdb_sesi "
    "within the Houdini installation at: ${HOUDINI_LIB_DIR}. "
  )
endif()

# Generate a new TestABIHoudini.cc source file from TestABI.cc,
# this one uses the Houdini OpenVDB headers and libs and lives in the build directory

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/TestABIHoudini.cc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/TestABI.cc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/TestABI.cc ${PROJECT_BINARY_DIR}/TestABIHoudini.cc
)

# Create an object library from the generated TestABIHoudini.cc file
# and set the include directories and add a HOUDINI variable

add_library(TestABIHoudini OBJECT ${PROJECT_BINARY_DIR}/TestABIHoudini.cc)
target_include_directories(TestABIHoudini SYSTEM PUBLIC ${HOUDINI_INCLUDE_DIR})
target_compile_options(TestABIHoudini PRIVATE -DHOUDINI)

# set_property(
#   SOURCE ${PROJECT_BINARY_DIR}/TestABIHoudini.cc
#   APPEND PROPERTY COMPILE_DEFINITIONS "HOUDINI;"
# )

# Create the binary from this object library and the main object file

add_executable(openvdb_houdini_abi_test
  TestABI.cc
  $<TARGET_OBJECTS:TestABIHoudini>
  main.cc
)

# Link against both versions of OpenVDB (libopenvdb and libopenvdb_sesi)

# Collect lib dependencies

if(NOT OPENVDB_BUILD_CORE)
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

target_link_libraries(openvdb_houdini_abi_test PUBLIC
  ${OPENVDB_LIB}
  ${OPENVDB_SESI_LIB}
)

if(SESI_HALF)
  target_link_libraries(openvdb_houdini_abi_test PUBLIC ${SESI_HALF})
endif()

add_test(vdb_abi_test openvdb_houdini_abi_test)
