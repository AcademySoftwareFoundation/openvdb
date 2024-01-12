# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

    CMake Configuration for VDB Example Fluids

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
project(VDBExampleFluids LANGUAGES CXX)

include(GNUInstallDirs)

set(SOURCE_FILES main.cc)
add_executable(vdb_example_fluids ${SOURCE_FILES})
target_link_libraries(vdb_example_fluids ${OPENVDB_BINARIES_DEPENDENT_LIBS})

install(TARGETS vdb_example_fluids RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
