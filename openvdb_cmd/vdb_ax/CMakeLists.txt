# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for VDB AX Binary

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(VDBAX LANGUAGES CXX)

include(GNUInstallDirs)

if(NOT OPENVDB_BUILD_AX)
  set(OPENVDBAX_LIB OpenVDB::openvdb_ax)
else()
  set(OPENVDBAX_LIB openvdb_ax)
endif()

set(OPENVDB_BINARIES_DEPENDENT_LIBS
  ${OPENVDBAX_LIB}
)

if(CONCURRENT_MALLOC STREQUAL "Jemalloc")
  find_package(Jemalloc REQUIRED)
  list(APPEND OPENVDB_BINARIES_DEPENDENT_LIBS Jemalloc::jemalloc)
elseif(CONCURRENT_MALLOC STREQUAL "Tbbmalloc")
  find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbbmalloc)
  list(APPEND OPENVDB_BINARIES_DEPENDENT_LIBS TBB::tbbmalloc)
endif()

set(SOURCE_FILES main.cc)
add_executable(vdb_ax ${SOURCE_FILES})
target_link_libraries(vdb_ax ${OPENVDB_BINARIES_DEPENDENT_LIBS})

install(TARGETS vdb_ax RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
