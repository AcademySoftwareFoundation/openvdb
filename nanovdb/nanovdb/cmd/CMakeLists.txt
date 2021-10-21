# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Command Line Tools

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(NanoVDBExamples LANGUAGES CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "--------- Configuring NanoVDB Cmd Tools ------------")
message(STATUS "----------------------------------------------------")

###############################################################################
# -----------------------------------------------------------------------------
# workaround for win32 bug when nvcc "--keep" is used.
if(WIN32 AND NANOVDB_CUDA_KEEP_PTX)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
endif()

set(NANOVDB_LOCAL_BINDIR ${PROJECT_BINARY_DIR}/bin)

# -----------------------------------------------------------------------------
# nanovdb_convert
if(NOT NANOVDB_USE_OPENVDB)
  message(WARNING " - OpenVDB required to build nanovdb_convert. Skipping.")
else()
  add_executable(nanovdb_convert convert/nanovdb_convert.cpp)
  target_link_libraries(nanovdb_convert PRIVATE nanovdb)
  install(TARGETS nanovdb_convert DESTINATION ${NANOVDB_LOCAL_BINDIR})
endif()

# -----------------------------------------------------------------------------
# nanovdb_print
add_executable(nanovdb_print print/nanovdb_print.cpp)
target_link_libraries(nanovdb_print PRIVATE nanovdb)
install(TARGETS nanovdb_print DESTINATION ${NANOVDB_LOCAL_BINDIR})

# -----------------------------------------------------------------------------
# nanovdb_validate
add_executable(nanovdb_validate validate/nanovdb_validate.cpp)
target_link_libraries(nanovdb_validate PRIVATE nanovdb)
install(TARGETS nanovdb_validate DESTINATION ${NANOVDB_LOCAL_BINDIR})
