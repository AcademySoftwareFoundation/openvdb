# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB AX Binaries

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
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

if(OPENVDB_ENABLE_RPATH)
  set(RPATHS "")
  # @todo There is probably a better way to do this for imported targets
  list(APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${IlmBase_LIBRARY_DIRS}
    ${Log4cplus_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
    ${Tbb_LIBRARY_DIRS}
  )
  if(OPENVDB_BUILD_AX)
    list(APPEND RPATHS ${CMAKE_INSTALL_FULL_LIBDIR})
  else()
    list(APPEND RPATHS ${OpenVDB_LIBRARY_DIRS})
  endif()

  list(REMOVE_DUPLICATES RPATHS)
  set_target_properties(vdb_ax
    PROPERTIES INSTALL_RPATH "${RPATHS}"
  )
  unset(RPATHS)
endif()

install(TARGETS vdb_ax RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
