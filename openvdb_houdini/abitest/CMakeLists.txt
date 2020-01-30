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

# Disable some compiler warnings in TBB and Half

add_compile_options(
  -Wno-conversion
  -Wno-shorten-64-to-32
)

# Create an object library that uses the Houdini OpenVDB headers

add_library(TestABI OBJECT TestABI.cc)
target_include_directories(TestABI PUBLIC ${OPENVDB_INCLUDEDIR})

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
