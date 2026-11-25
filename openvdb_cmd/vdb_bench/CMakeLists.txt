# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
#[=======================================================================[

  CMake Configuration for VDB Bench

#]=======================================================================]

cmake_minimum_required(VERSION 3.24)
project(VDBBench LANGUAGES CXX)

include(GNUInstallDirs)

# One file per tool group. Explicit instantiation can be off, so the tool
# templates are compiled in these files, and code added to a file changes what
# the compiler inlines in the rest of it.
set(SOURCE_FILES
  main.cc
  LevelSet.cc
  Analysis.cc
  Operators.cc
  Filter.cc
  Poisson.cc
  Sampling.cc
  Meshing.cc
  Accessor.cc
  Summary.cc)
add_executable(vdb_bench ${SOURCE_FILES})
target_link_libraries(vdb_bench ${OPENVDB_BINARIES_DEPENDENT_LIBS})
