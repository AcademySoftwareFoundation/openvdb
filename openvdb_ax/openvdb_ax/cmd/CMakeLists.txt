# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB AX Binaries

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(OpenVDBAXBinaries LANGUAGES CXX)

include(GNUInstallDirs)

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "---------- Configuring OpenVDBAXBinaries -----------")
message(STATUS "----------------------------------------------------")

##########################################################################

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

set(VDB_AX_SOURCE_FILES openvdb_ax.cc)
add_executable(vdb_ax ${VDB_AX_SOURCE_FILES})
target_link_libraries(vdb_ax ${OPENVDB_BINARIES_DEPENDENT_LIBS})

install(TARGETS vdb_ax RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
