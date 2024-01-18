# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for the OpenVDB Houdini Plugin.

  See the OpenVDBHoudiniSetup.cmake module for more information on how to
  locate and use a Houdini Installation.

  There are two main stages for the OpenVDB Houdini build; the OpenVDB
  Houdini base shared library and all subsequent dependent nodes (SOPs,
  SHOPs, procedurals etc.) The default install looks as follows:

    <prefix>/bin           - Windows runtime components
    <prefix>/include       - Shared openvdb_houdini headers
    <prefix>/config/Icons  - OpenVDB Icon files
    <prefix>/{lib|lib64}   - Location of libopenvdb_houdini
    <prefix>/python2.7libs - python.rc for Houdini startup
    <prefix>/scripts/sop   - Relevant SOP scripts
    <prefix>/dso           - Location of DSOs
    <prefix>/help          - Individual help files

  By default, the prefix is set to the value returned by the Houdini
  CMake configuration. This is the default installation path for all
  Houdini plugins. You can disable this behavior by setting the following
  variable to OFF:
    USE_DEFAULT_HOUDINI_INSTALL

  CMake will instead read from CMAKE_INSTALL_PREFIX. If you're building
  multiple components of the OpenVDB repository and wish to customize the
  prefix, the following cached variable can be set:
    OPENVDB_HOUDINI_INSTALL_PREFIX

  This allows you to use CMAKE_INSTALL_PREFIX for the core library
  location and a custom location for the Houdini plugin.

  Note: Individual DSO components can be turned off with their own CMake
  variable "OPENVDB_BUILD_{NAME}" i.e. OPENVDB_BUILD_SOP_OpenVDB_Scatter=OFF

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
project(OpenVDBHoudini LANGUAGES CXX)


option(USE_DEFAULT_HOUDINI_INSTALL [=[
Use the default install location returned by the Houdini CMake function,
houdini_get_default_install_dir():
  Linux: $ENV{HOME}/houdiniX.X
  Mac: $ENV{HOME}/Library/Preferences/houdini/X.X
  Windows: $ENV{HOMEDRIVE}$ENV{HOMEPATH}\Documents\houdiniX.X
If OFF, CMAKE_INSTALL_PREFIX is used.]=] ON)
option(OPENVDB_INSTALL_HOUDINI_PYTHONRC [=[Install a Houdini startup script that sets
the visibilty of OpenVDB nodes and their native equivalents.]=] OFF)

set(_OPENVDB_HOUDINI_OPHIDE_POLICY_OPTIONS none aswf native)
if(NOT OPENVDB_HOUDINI_OPHIDE_POLICY)
  set(OPENVDB_HOUDINI_OPHIDE_POLICY none CACHE STRING
    "Choose an ophide policy, options are: none aswf native. If
    OPENVDB_HOUDINI_OPHIDE_POLICY is set to 'aswf', hide AWSF SOPs for which a
    native Houdini equivalent exists. If $OPENVDB_OPHIDE_POLICY is set to
    'native', hide native Houdini SOPs for which an ASWF equivalent exists.
    Otherwise show both SOPs (the default). Note that the environment variable
    OPENVDB_OPHIDE_POLICY can be used to override this flag at runtime." FORCE
  )
else()
  string(TOLOWER ${OPENVDB_HOUDINI_OPHIDE_POLICY} OPENVDB_HOUDINI_OPHIDE_POLICY)
  if(NOT ${OPENVDB_HOUDINI_OPHIDE_POLICY} IN_LIST _OPENVDB_HOUDINI_OPHIDE_POLICY_OPTIONS)
    message(WARNING "Unrecognized value for OPENVDB_HOUDINI_OPHIDE_POLICY, using none instead.")
    set(OPENVDB_HOUDINI_OPHIDE_POLICY none CACHE STRING FORCE)
  endif()
endif()

set(OPENVDB_DSO_NAMES "" CACHE STRING [=[
  An optional list of Houdini DSOs to build. If not provided, all Houdini DSOs will be built.
  DSOs should be listed without the .cc suffix, for example the string
  \"SOP_OpenVDB_Activate;SOP_OpenVDB_Resample\" would build just the VDB Activate and
  VDB Resample SOPs.]=])

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "------------ Configuring OpenVDBHoudini ------------")
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
message(STATUS "Houdini base install path: ${OPENVDB_HOUDINI_INSTALL_PREFIX}")

# Install paths for bin/lib/include components
set(OPENVDB_HOUDINI_BIN_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
  CACHE PATH "Install path for the OpenVDB Houdini shared library")
set(OPENVDB_HOUDINI_LIB_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
  CACHE PATH "Install path for the OpenVDB Houdini shared library")
set(OPENVDB_HOUDINI_INCLUDE_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
  CACHE PATH "Install path for the OpenVDB Houdini headers")

# Install paths for dsos and scripts
set(OPENVDB_HOUDINI_DSO_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/dso
  CACHE PATH "Install path for the OpenVDB Houdini dsos")
set(OPENVDB_HOUDINI_MANTRA_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/dso/mantra
  CACHE PATH "Install path for the OpenVDB Houdini mantra nodes")
set(OPENVDB_HOUDINI_SOP_SCRIPT_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/scripts/sop
  CACHE PATH "Install path for the OpenVDB Houdini node scripts")
set(OPENVDB_HOUDINI_HELP_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/help
  CACHE PATH "Install path for the OpenVDB Houdini help files")
set(OPENVDB_HOUDINI_ICON_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/config/Icons
  CACHE PATH "Install path for the OpenVDB Houdini node icons")
set(OPENVDB_HOUDINI_PYTHON_INSTALL_PREFIX ${OPENVDB_HOUDINI_INSTALL_PREFIX}/python2.7libs
  CACHE PATH "Install path for the OpenVDB Houdini startup script")

#########################################################################

# Collect and configure lib dependencies

if(NOT OPENVDB_BUILD_CORE)
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

# @note Linking openvdb_houdini against Houdini brings in all interface compile
# definitions, including MAKING_DSO

set(OPENVDB_HOUDINI_DEPENDENT_LIBS
  ${OPENVDB_LIB}
  Houdini
)

#########################################################################

# Copy the files belonging to houdini_utils

set(OPENVDB_HOUDINI_UTILS_DIR ${PROJECT_BINARY_DIR}/houdini_utils)
file(MAKE_DIRECTORY ${OPENVDB_HOUDINI_UTILS_DIR})
file(COPY
    geometry.h
    ParmFactory.h
  DESTINATION
    ${OPENVDB_HOUDINI_UTILS_DIR}
)

# Copy the files belonging to openvdb_houdini
set(OPENVDB_HOUDINI_LOCAL_DIR ${PROJECT_BINARY_DIR}/openvdb_houdini)
file(MAKE_DIRECTORY ${OPENVDB_HOUDINI_LOCAL_DIR})
file(COPY
    AttributeTransferUtil.h
    GeometryUtil.h
    GT_GEOPrimCollectVDB.h
    GU_VDBPointTools.h
    PointUtils.h
    SOP_NodeVDB.h
    SOP_VDBVerbUtils.h
    UT_VDBTools.h
    UT_VDBUtils.h
    Utils.h
  DESTINATION
    ${OPENVDB_HOUDINI_LOCAL_DIR}
)

add_library(openvdb_houdini SHARED
  GEO_VDBTranslator.cc
  geometry.cc
  GeometryUtil.cc
  GT_GEOPrimCollectVDB.cc
  GU_VDBPointTools.cc
  ParmFactory.cc
  PointUtils.cc
  SOP_NodeVDB.cc
  UT_VDBUtils.cc
  Utils.cc
)

target_include_directories(openvdb_houdini PUBLIC
  ${OPENVDB_HOUDINI_UTILS_DIR}
  ${OPENVDB_HOUDINI_LOCAL_DIR}
  ${PROJECT_BINARY_DIR}
)

target_link_libraries(openvdb_houdini PUBLIC
  ${OPENVDB_HOUDINI_DEPENDENT_LIBS}
)

target_compile_definitions(openvdb_houdini PRIVATE "-DOPENVDB_HOUDINI_PRIVATE")

set_target_properties(openvdb_houdini
  PROPERTIES
    OUTPUT_NAME openvdb_houdini
    SOVERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}
    VERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION}
)

if (OPENVDB_HOUDINI_OPHIDE_POLICY AND NOT ${OPENVDB_HOUDINI_OPHIDE_POLICY} STREQUAL "none")
  target_compile_definitions(openvdb_houdini PRIVATE
    "-DOPENVDB_OPHIDE_POLICY=${OPENVDB_HOUDINI_OPHIDE_POLICY}")
endif()

# If OPENVDB_BUILD_SOP_OpenVDB_AX has not been explicitly set,
# see if we can enabled the OpenVDB AX SOP
if(NOT DEFINED OPENVDB_BUILD_SOP_OpenVDB_AX)
  if(USE_AX)
    set(OPENVDB_BUILD_SOP_OpenVDB_AX ON)
  elseif(OPENVDB_DSO_NAMES)
    if(SOP_OpenVDB_AX IN_LIST OPENVDB_DSO_NAMES)
      set(OPENVDB_BUILD_SOP_OpenVDB_AX ON)
    endif()
  endif()
endif()

if(NOT OPENVDB_DSO_NAMES)
  list(APPEND OPENVDB_DSO_NAMES
    GR_PrimVDBPoints
    SHOP_OpenVDB_Points
    SOP_OpenVDB_Activate
    SOP_OpenVDB_Advect
    SOP_OpenVDB_Advect_Points
    SOP_OpenVDB_Analysis
    SOP_OpenVDB_Clip
    SOP_OpenVDB_Combine
    SOP_OpenVDB_Convert
    SOP_OpenVDB_Create
    SOP_OpenVDB_Densify
    SOP_OpenVDB_Diagnostics
    SOP_OpenVDB_Extrapolate
    SOP_OpenVDB_Fill
    SOP_OpenVDB_Filter
    SOP_OpenVDB_Filter_Level_Set
    SOP_OpenVDB_Fracture
    SOP_OpenVDB_From_Particles
    SOP_OpenVDB_From_Polygons
    SOP_OpenVDB_LOD
    SOP_OpenVDB_Merge
    SOP_OpenVDB_Metadata
    SOP_OpenVDB_Morph_Level_Set
    SOP_OpenVDB_Noise
    SOP_OpenVDB_Occlusion_Mask
    SOP_OpenVDB_Platonic
    SOP_OpenVDB_Points_Convert
    SOP_OpenVDB_Points_Delete
    SOP_OpenVDB_Points_Group
    SOP_OpenVDB_Potential_Flow
    SOP_OpenVDB_Prune
    SOP_OpenVDB_Rasterize_Frustum
    SOP_OpenVDB_Rasterize_Points
    SOP_OpenVDB_Ray
    SOP_OpenVDB_Read
    SOP_OpenVDB_Rebuild_Level_Set
    SOP_OpenVDB_Remap
    SOP_OpenVDB_Remove_Divergence
    SOP_OpenVDB_Resample
    SOP_OpenVDB_Sample_Points
    SOP_OpenVDB_Scatter
    SOP_OpenVDB_Segment
    SOP_OpenVDB_Sort_Points
    SOP_OpenVDB_To_Polygons
    SOP_OpenVDB_To_Spheres
    SOP_OpenVDB_Topology_To_Level_Set
    SOP_OpenVDB_Transform
    SOP_OpenVDB_Vector_Merge
    SOP_OpenVDB_Vector_Split
    SOP_OpenVDB_Visualize
    SOP_OpenVDB_Write
    VRAY_OpenVDB_Points)

  if(OPENVDB_BUILD_SOP_OpenVDB_AX)
    list(APPEND OPENVDB_DSO_NAMES SOP_OpenVDB_AX)
  endif()
endif()

add_custom_target(openvdb_houdini_dsos
  COMMENT "Group target which corresponds to all Houdini Nodes and DSOs "
  "which aren't part of the shared library i.e. all targets in OPENVDB_DSO_NAMES."
)

# Add all dsos as libraries and configure them for houdini

message(STATUS "Configuring Houdini DSOs...")

foreach(DSO_NAME ${OPENVDB_DSO_NAMES})
  # Allow individual DSOs to be disabled through their own CMake variable
  # "OPENVDB_BUILD_{NAME}" i.e. OPENVDB_BUILD_SOP_OpenVDB_Scatter=OFF

  if(DEFINED OPENVDB_BUILD_${DSO_NAME})
    if(NOT OPENVDB_BUILD_${DSO_NAME})
      continue()
    endif()
  endif()

  add_library(${DSO_NAME} SHARED ${DSO_NAME}.cc)
  target_link_libraries(${DSO_NAME} PUBLIC openvdb_houdini)

  if(OPENVDB_ENABLE_RPATH)
    set_target_properties(${DSO_NAME}
      PROPERTIES INSTALL_RPATH
        "${CMAKE_INSTALL_RPATH};${OPENVDB_HOUDINI_BIN_INSTALL_PREFIX};${OPENVDB_HOUDINI_LIB_INSTALL_PREFIX}")
  endif()

  # Call houdini_configure_target to setup the sesi tag information. We set
  # INSTDIR to the project build directory and configure the actuall install
  # paths ourselves (otherwise cmake will configure the build to build directly
  # to the install dir)

  houdini_configure_target(${DSO_NAME}
    INSTDIR ${PROJECT_BINARY_DIR}
    PREFIX ""
    TAGINFO ""
  )

  add_dependencies(openvdb_houdini_dsos ${DSO_NAME})
endforeach()

if(OPENVDB_BUILD_SOP_OpenVDB_AX)
  # Link the AX SOP against openvdb_ax
  if(NOT OPENVDB_BUILD_AX)
    find_package(OpenVDB REQUIRED openvdb_ax)
    target_link_libraries(SOP_OpenVDB_AX PUBLIC OpenVDB::openvdb_ax)
  else()
    target_link_libraries(SOP_OpenVDB_AX PUBLIC openvdb_ax)
  endif()
endif()

# These defines are in Platform.h when building against 8.1 and later
if(OpenVDB_VERSION VERSION_LESS 8.1.0)
  # This is only required for the OpenVDB Write SOP which enables a compression menu
  # option if blosc and zlib are supported.
  if(USE_BLOSC OR OpenVDB_USES_BLOSC)
    if(TARGET SOP_OpenVDB_Write)
      target_compile_definitions(SOP_OpenVDB_Write PRIVATE "-DOPENVDB_USE_BLOSC")
    endif()
  endif()
  if(USE_BLOSC OR OpenVDB_USES_BLOSC OR USE_ZLIB OR OpenVDB_USES_ZLIB)
    if(TARGET SOP_OpenVDB_Write)
      target_compile_definitions(SOP_OpenVDB_Write PRIVATE "-DOPENVDB_USE_ZLIB")
    endif()
  endif()
endif()

#

##########
# Installs
##########

install(DIRECTORY
  ${OPENVDB_HOUDINI_UTILS_DIR}
  ${OPENVDB_HOUDINI_LOCAL_DIR}
  DESTINATION ${OPENVDB_HOUDINI_INCLUDE_INSTALL_PREFIX}
)

install(TARGETS openvdb_houdini
  RUNTIME DESTINATION ${OPENVDB_HOUDINI_BIN_INSTALL_PREFIX}
  LIBRARY DESTINATION ${OPENVDB_HOUDINI_LIB_INSTALL_PREFIX}
  ARCHIVE DESTINATION ${OPENVDB_HOUDINI_LIB_INSTALL_PREFIX}
)

foreach(DSO_NAME ${OPENVDB_DSO_NAMES})
  if(DSO_NAME MATCHES "VRAY_*")
    install(TARGETS
      ${DSO_NAME}
      DESTINATION ${OPENVDB_HOUDINI_MANTRA_INSTALL_PREFIX}
    )
  else()
    install(TARGETS
      ${DSO_NAME}
      DESTINATION ${OPENVDB_HOUDINI_DSO_INSTALL_PREFIX}
    )
  endif()
endforeach()

install(FILES
  SOP_OpenVDB.svg
  DESTINATION ${OPENVDB_HOUDINI_ICON_INSTALL_PREFIX}
)

install(FILES
  DW_OpenVDBRasterizePoints.cmd
  DESTINATION ${OPENVDB_HOUDINI_SOP_SCRIPT_INSTALL_PREFIX}
)

install(DIRECTORY help/
  DESTINATION ${OPENVDB_HOUDINI_HELP_INSTALL_PREFIX})

if(OPENVDB_INSTALL_HOUDINI_PYTHONRC)
  install(FILES
    pythonrc.py
    DESTINATION ${OPENVDB_HOUDINI_PYTHON_INSTALL_PREFIX}
  )
endif()
