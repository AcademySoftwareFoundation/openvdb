# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for VDB Print

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(VDBPrint LANGUAGES CXX)

include(GNUInstallDirs)

set(SOURCE_FILES main.cc)
add_executable(vdb_print ${SOURCE_FILES})
target_link_libraries(vdb_print ${OPENVDB_BINARIES_DEPENDENT_LIBS})

install(TARGETS vdb_print RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
