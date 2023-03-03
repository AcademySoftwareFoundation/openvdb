# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Command Line Tools

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
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

# -----------------------------------------------------------------------------

if(NOT NANOVDB_USE_OPENVDB)
  message(WARNING " - OpenVDB required to build nanovdb_convert. Skipping.")
else()
  add_executable(nanovdb_convert convert/nanovdb_convert.cc)
  target_link_libraries(nanovdb_convert PRIVATE nanovdb)
  install(TARGETS nanovdb_convert RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# -----------------------------------------------------------------------------
# nanovdb_print
add_executable(nanovdb_print print/nanovdb_print.cc)
target_link_libraries(nanovdb_print PRIVATE nanovdb)
install(TARGETS nanovdb_print RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# -----------------------------------------------------------------------------
# nanovdb_validate
add_executable(nanovdb_validate validate/nanovdb_validate.cc)
target_link_libraries(nanovdb_validate PRIVATE nanovdb)
install(TARGETS nanovdb_validate RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
