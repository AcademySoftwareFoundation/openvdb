# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for the OpenVDB AX Houdini Plugin.

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(OpenVDBAXHoudini LANGUAGES CXX)

option(USE_DEFAULT_HOUDINI_INSTALL [=[
Use the default install location returned by the Houdini CMake function,
houdini_get_default_install_dir():
  Linux: $ENV{HOME}/houdiniX.X
  Mac: $ENV{HOME}/Library/Preferences/houdini/X.X
  Windows: $ENV{HOMEDRIVE}$ENV{HOMEPATH}\Documents\houdiniX.X
If OFF, CMAKE_INSTALL_PREFIX is used.]=] ON)

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "---------- Configuring OpenVDBAXHoudini ----------")
message(STATUS "----------------------------------------------------")

#########################################################################

# Configure prefix

if(NOT OPENVDB_HOUDINI_INSTALL_PREFIX)
  if(USE_DEFAULT_HOUDINI_INSTALL)
    # Function from the Houdini CMake which is brought in by OpenVDBHoudiniSetup.cmake
    houdini_get_default_install_dir( HOUDINI_INSTALL_LOCATION)
  else()
    set(HOUDINI_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX})
  endif()
endif()

set(OPENVDB_HOUDINI_INSTALL_PREFIX ${HOUDINI_INSTALL_LOCATION}
  CACHE PATH "Base install path for OpenVDB Houdini shared libraries, dsos and headers."
)
set(OPENVDB_HOUDINI_DSODIR "${OPENVDB_HOUDINI_INSTALL_PREFIX}/dso")
set(OPENVDB_HOUDINI_INCLUDEDIR "${OPENVDB_HOUDINI_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")

#########################################################################

# This module relies on 3 libs; openvdb, openvdb_ax and openvdb_houdini.
# if OPENVDB_BUILD_CORE is ON, all deps must also be on
# openvdb and openvdb_houdini must be either both enabled or disabled.
# openvdb and openvdb_ax must be either both enabled or disabled.

if(OPENVDB_BUILD_CORE)
  if(NOT OPENVDB_BUILD_HOUDINI_PLUGIN OR NOT OPENVDB_BUILD_AX)
    message(FATAL_ERROR "Invalid CMake build configuration. OPENVDB_BUILD_CORE is ON, "
      "but OPENVDB_BUILD_HOUDINI_PLUGIN or OPENVDB_BUILD_AX is OFF. If rebuilding the "
      "core library, sub components must also be rebuilt.")
  endif()
endif()

# Get OpenVDB AX Houdini lib deps

set(LIB_DEPS "")

if(OPENVDB_BUILD_CORE)
  list(APPEND LIB_DEPS openvdb openvdb_ax openvdb_houdini)
else()
  list(APPEND LIB_DEPS OpenVDB::openvdb)
  if(NOT OPENVDB_BUILD_HOUDINI_PLUGIN)
    find_package(OpenVDB REQUIRED openvdb_houdini)
    list(APPEND LIB_DEPS OpenVDB::openvdb_houdini)
  else()
    list(APPEND LIB_DEPS openvdb_houdini)
  endif()
  if(NOT OPENVDB_BUILD_AX)
    find_package(OpenVDB REQUIRED openvdb_ax)
    list(APPEND LIB_DEPS OpenVDB::openvdb_ax)
  else()
    list(APPEND LIB_DEPS openvdb_ax)
  endif()
endif()

#### ADD TARGET

add_library(SOP_OpenVDB_AX SHARED SOP_OpenVDB_AX.cc)
set_target_properties(SOP_OpenVDB_AX PROPERTIES PREFIX "")

target_include_directories(SOP_OpenVDB_AX PRIVATE . )
target_link_libraries(SOP_OpenVDB_AX PUBLIC ${LIB_DEPS} Houdini)

if(OPENVDB_ENABLE_RPATH)
  set(RPATHS)
  if(OPENVDB_BUILD_AX)
    list(APPEND RPATHS ${CMAKE_INSTALL_PREFIX}/lib)
  else()
    list(APPEND RPATHS ${OpenVDB_LIBRARY_DIRS})
  endif()

  list(REMOVE_DUPLICATES RPATHS)
  set_target_properties(SOP_OpenVDB_AX
    PROPERTIES INSTALL_RPATH "${RPATHS}"
  )
endif()

##### INSTALL

set(OPENVDB_AX_HOUDINI_INCLUDE_FILES ax/HoudiniAXUtils.h)

install(TARGETS SOP_OpenVDB_AX
  DESTINATION
    ${OPENVDB_HOUDINI_DSODIR}
)

install(FILES ${OPENVDB_AX_HOUDINI_INCLUDE_FILES}
  DESTINATION
    ${OPENVDB_HOUDINI_INCLUDEDIR}/openvdb_ax_houdini/ax
)
