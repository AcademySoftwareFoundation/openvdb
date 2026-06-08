# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
#[=======================================================================[

  CMake Configuration for VDB Fillet

#]=======================================================================]

cmake_minimum_required(VERSION 3.24)
project(VDBFillet LANGUAGES CXX)

include(GNUInstallDirs)

set(SOURCE_FILES main.cc)
add_executable(vdb_fillet ${SOURCE_FILES})
target_link_libraries(vdb_fillet ${OPENVDB_BINARIES_DEPENDENT_LIBS})

install(TARGETS vdb_fillet RUNTIME DESTINATION ${OPENVDB_INSTALL_BINDIR})
