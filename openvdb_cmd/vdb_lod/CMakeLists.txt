# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for VDB LOD

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(VDBLOD LANGUAGES CXX)

include(GNUInstallDirs)

set(SOURCE_FILES main.cc)
add_executable(vdb_lod ${SOURCE_FILES})
target_link_libraries(vdb_lod ${OPENVDB_BINARIES_DEPENDENT_LIBS})

install(TARGETS vdb_lod RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
