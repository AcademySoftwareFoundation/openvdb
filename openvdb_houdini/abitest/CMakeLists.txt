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

cmake_minimum_required(VERSION 3.3)
project(OpenVDBHoudiniABITest)

# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "-------- Configuring OpenVDBHoudiniABITest ---------")
message(STATUS "----------------------------------------------------")

##########################################################################

# This binary is composed of two object files:
# - TestABI uses headers from this version of OpenVDB
# - main uses headers from the version of OpenVDB shipped with Houdini

if(NOT DEFINED _HOUDINI_INCLUDE_DIR OR NOT DEFINED _HOUDINI_LIB_DIR)
  message(FATAL_ERROR "The Houdini ABI Test requires the Houdini include "
    "directory and lib directory variables to be defined"
  )
endif()

# Find libopenvdb_sesi library
find_library(OPENVDB_SESI_LIB openvdb_sesi
  ${_FIND_HOUDINI_ADDITIONAL_OPTIONS}
  PATHS ${_HOUDINI_LIB_DIR}
)

if(NOT EXISTS ${OPENVDB_SESI_LIB})
  message(FATAL_ERROR "The Houdini ABI Test is unable to locate libopenvdb_sesi "
    "within the Houdini installation at: ${_HOUDINI_LIB_DIR}. "
  )
endif()

# Create an object library that uses the Houdini OpenVDB headers

add_library(TestABI OBJECT TestABI.cc)
target_include_directories(TestABI SYSTEM PUBLIC ${_HOUDINI_INCLUDE_DIR})

# Create the binary from this object library and the main object file

add_executable(openvdb_houdini_abi_test
  $<TARGET_OBJECTS:TestABI>
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

add_test(vdb_abi_test openvdb_houdini_abi_test)
