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
add_library(vdb_print_obj OBJECT ${SOURCE_FILES})

# Propagate interface include directories from the dependencies to the object library

foreach(DEP ${OPENVDB_BINARIES_DEPENDENT_LIBS})
  target_include_directories(vdb_print_obj PUBLIC $<TARGET_PROPERTY:${DEP},INTERFACE_INCLUDE_DIRECTORIES>)
endforeach()

add_executable(vdb_print $<TARGET_OBJECTS:vdb_print_obj>)

target_link_libraries(vdb_print ${OPENVDB_BINARIES_DEPENDENT_LIBS})

install(TARGETS vdb_print RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
