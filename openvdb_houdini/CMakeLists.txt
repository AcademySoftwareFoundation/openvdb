# Copyright (c) 2012-2019 DreamWorks Animation LLC
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DreamWorks Animation nor the names of
# its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
# LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
#
#[=======================================================================[

  CMake Configuration for the OpenVDB Houdini Plugin.

  See the OpenVDBHoudiniSetup.cmake module for more information on how to
  locate and use a Houdini Installation.

  There are two main stages for the OpenVDB Houdini build; the OpenVDB
  Houdini base shared library and all subsequent dependent nodes (SOPs,
  SHOPs, procedurals etc.) Both are installed to the same prefix:

    <prefix>/include  - Shared openvdb_houdini headers
    <prefix>/lib      - Location of libopenvdb_houdini
    <prefix>/dso      - Location of all other .so and required files
                        such as icons and scripts

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

#]=======================================================================]

PROJECT ( OpenVDBHoudini )
CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )
IF ( POLICY CMP0074 )
  CMAKE_POLICY ( SET CMP0074 NEW )
ENDIF ()

OPTION ( USE_DEFAULT_HOUDINI_INSTALL [=[
Use the default install location returned by the Houdini CMake function,
houdini_get_default_install_dir():
  Linux: $ENV{HOME}/houdiniX.X
  Mac: $ENV{HOME}/Library/Preferences/houdini/X.X
  Windows: $ENV{HOMEDRIVE}$ENV{HOMEPATH}\Documents\houdiniX.X
If OFF, CMAKE_INSTALL_PREFIX is used.]=] ON )

# SESITAG CMake options
OPTION ( OPENVDB_HOUDINI_ENABLE_SESI_TAG [=[
Enable SESI tag information on the OpenVDB DSOs. This is legacy functionality for
older Houdini versions and will be removed in Houdini 18. ]=] OFF )

MARK_AS_ADVANCED (
  OPENVDB_HOUDINI_ENABLE_SESI_TAG
  )

#########################################################################

MESSAGE ( STATUS "----------------------------------------------------" )
MESSAGE ( STATUS "------------ Configuring OpenVDBHoudini ------------" )
MESSAGE ( STATUS "----------------------------------------------------" )

#########################################################################

# Configure prefix

IF ( NOT OPENVDB_HOUDINI_INSTALL_PREFIX )
  IF ( USE_DEFAULT_HOUDINI_INSTALL )
    # Function from the Houdini CMake which is brought in by OpenVDBHoudiniSetup.cmake
    houdini_get_default_install_dir( HOUDINI_INSTALL_LOCATION )
  ELSE ()
    SET ( HOUDINI_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX} )
  ENDIF ()
ENDIF ()

SET ( OPENVDB_HOUDINI_INSTALL_PREFIX ${HOUDINI_INSTALL_LOCATION}
  CACHE PATH "Base install path for OpenVDB Houdini shared libraries, dsos and headers."
  )

MESSAGE ( STATUS "Houdini base install path: ${OPENVDB_HOUDINI_INSTALL_PREFIX}" )

#########################################################################

# Collect and configure lib dependencies

IF ( NOT OPENVDB_BUILD_CORE )
  SET ( OPENVDB_LIB OpenVDB::openvdb )
ELSE ()
  SET ( OPENVDB_LIB openvdb )
ENDIF ()

# @note Linking openvdb_houdini against Houdini brings in all interface compile
# definitions, including MAKING_DSO

SET ( OPENVDB_HOUDINI_DEPENDENT_LIBS
  ${OPENVDB_LIB}
  Houdini
)

#########################################################################

# Copy the files belonging to houdini_utils

SET ( OPENVDB_HOUDINI_UTILS_DIR ${PROJECT_BINARY_DIR}/houdini_utils )
FILE ( MAKE_DIRECTORY ${OPENVDB_HOUDINI_UTILS_DIR} )
FILE ( COPY
    houdini/geometry.h
    houdini/OP_NodeChain.h
    houdini/ParmFactory.h
  DESTINATION
    ${OPENVDB_HOUDINI_UTILS_DIR}
  )

# Copy the files belonging to openvdb_houdini
SET  ( OPENVDB_HOUDINI_LOCAL_DIR ${PROJECT_BINARY_DIR}/openvdb_houdini )
FILE ( MAKE_DIRECTORY ${OPENVDB_HOUDINI_LOCAL_DIR} )
FILE ( COPY
    houdini/AttributeTransferUtil.h
    houdini/GEO_PrimVDB.h
    houdini/GeometryUtil.h
    houdini/GT_GEOPrimCollectVDB.h
    houdini/GU_PrimVDB.h
    houdini/GU_VDBPointTools.h
    houdini/PointUtils.h
    houdini/SOP_NodeVDB.h
    houdini/SOP_VDBVerbUtils.h
    houdini/UT_VDBTools.h
    houdini/UT_VDBUtils.h
    houdini/Utils.h
  DESTINATION
    ${OPENVDB_HOUDINI_LOCAL_DIR}
  )

ADD_LIBRARY ( openvdb_houdini SHARED
  houdini/GEO_PrimVDB.cc
  houdini/GEO_VDBTranslator.cc
  houdini/geometry.cc
  houdini/GeometryUtil.cc
  houdini/GT_GEOPrimCollectVDB.cc
  houdini/GU_PrimVDB.cc
  houdini/GU_VDBPointTools.cc
  houdini/ParmFactory.cc
  houdini/PointUtils.cc
  houdini/SOP_NodeVDB.cc
  houdini/UT_VDBUtils.cc
  houdini/Utils.cc
  )

TARGET_INCLUDE_DIRECTORIES ( openvdb_houdini PUBLIC
  ${OPENVDB_HOUDINI_UTILS_DIR}
  ${OPENVDB_HOUDINI_LOCAL_DIR}
  ${PROJECT_BINARY_DIR}
  )

TARGET_LINK_LIBRARIES ( openvdb_houdini PUBLIC
  ${OPENVDB_HOUDINI_DEPENDENT_LIBS}
  )

IF ( OPENVDB_ENABLE_RPATH )
  # @todo There is probably a better way to do this for imported targets
  SET ( RPATHS "" )
  LIST ( APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${IlmBase_LIBRARY_DIRS}
    ${Log4cplus_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
    ${Tbb_LIBRARY_DIRS}
  )
  IF ( OPENVDB_BUILD_CORE )
    LIST ( APPEND ${CMAKE_INSTALL_PREFIX}/lib )
  ELSE ()
    LIST ( APPEND ${OpenVDB_LIBRARY_DIRS} )
  ENDIF ()

  LIST ( REMOVE_DUPLICATES RPATHS )

  SET_TARGET_PROPERTIES ( openvdb_houdini
    PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  UNSET ( RPATHS )
ENDIF ()


SET_TARGET_PROPERTIES ( openvdb_houdini
  PROPERTIES
    OUTPUT_NAME openvdb_houdini
    SOVERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}
    VERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION}
  )

SET ( OPENVDB_DSO_NAMES
  GR_PrimVDBPoints
  SHOP_OpenVDB_Points
  SOP_OpenVDB_Advect
  SOP_OpenVDB_Advect_Points
  SOP_OpenVDB_Analysis
  SOP_OpenVDB_Clip
  SOP_OpenVDB_Combine
  SOP_OpenVDB_Convert
  SOP_OpenVDB_Create
  SOP_OpenVDB_Densify
  SOP_OpenVDB_Diagnostics
  SOP_OpenVDB_Fill
  SOP_OpenVDB_Filter
  SOP_OpenVDB_Filter_Level_Set
  SOP_OpenVDB_Fracture
  SOP_OpenVDB_From_Particles
  SOP_OpenVDB_From_Polygons
  SOP_OpenVDB_LOD
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
  VRAY_OpenVDB_Points
  )

ADD_CUSTOM_TARGET ( openvdb_houdini_dsos
  COMMENT "Group target which corresponds to all Houdini Nodes and DSOs "
  "which aren't part of the shared library i.e. all targets in OPENVDB_DSO_NAMES."
  )

# Add all dsos as libraries and configure them for houdini

MESSAGE ( STATUS "Configuring Houdini DSOs..." )

FOREACH ( DSO_NAME ${OPENVDB_DSO_NAMES} )

  ADD_LIBRARY ( ${DSO_NAME} SHARED houdini/${DSO_NAME}.cc )
  TARGET_LINK_LIBRARIES ( ${DSO_NAME} openvdb_houdini )

  # Call houdini_configure_target to setup the sesi tag information. We set
  # INSTDIR to the project build directory and configure the actuall install
  # paths ourselves (otherwise cmake will configure the build to build directly
  # to the install dir)

  houdini_configure_target ( ${DSO_NAME}
    INSTDIR ${PROJECT_BINARY_DIR}
    PREFIX ""
    TAGINFO ""
  )

  # If the sesi tag needs removing, manually configure the compile
  # options for each DSO and replace the UT_DSO_TAGINFO define

  IF ( NOT OPENVDB_HOUDINI_ENABLE_SESI_TAG )
    GET_TARGET_PROPERTY ( DSO_DEFINITIONS ${DSO_NAME} COMPILE_DEFINITIONS )
    STRING ( REGEX REPLACE "UT_DSO_TAGINFO=\".+\"" "" DSO_DEFINITIONS "${DSO_DEFINITIONS}")
    SET_TARGET_PROPERTIES ( ${DSO_NAME}
      PROPERTIES COMPILE_DEFINITIONS "${DSO_DEFINITIONS}"
    )
  ENDIF ()

  # Configure rpaths
  # Encode the path to libopenvdb.so into plugins, and because some
  # plugins might depend on Houdini libraries that are not linked into
  # hython (among other tools), encode the Houdini library path as well.

  IF ( OPENVDB_ENABLE_RPATH )
    SET ( RPATHS )
    IF ( OPENVDB_BUILD_CORE )
      LIST ( APPEND RPATHS ${CMAKE_INSTALL_PREFIX}/lib )
    ELSE ()
      LIST ( APPEND RPATHS ${OpenVDB_LIBRARY_DIRS} )
    ENDIF ()

    LIST ( APPEND RPATHS ${OPENVDB_HOUDINI_INSTALL_PREFIX}/lib )

    LIST ( REMOVE_DUPLICATES RPATHS )
    SET_TARGET_PROPERTIES ( ${DSO_NAME}
      PROPERTIES INSTALL_RPATH "${RPATHS}"
      )
  ENDIF ()

  ADD_DEPENDENCIES ( openvdb_houdini_dsos ${DSO_NAME} )

ENDFOREACH ()

# This is only required for the OpenVDB Write SOP which enables a blosc menu
# option if blosc is supported.

IF ( USE_BLOSC OR OpenVDB_USES_BLOSC )
  IF ( TARGET SOP_OpenVDB_Write )
    TARGET_COMPILE_DEFINITIONS ( SOP_OpenVDB_Write PRIVATE "-DOPENVDB_USE_BLOSC" )
  ENDIF ()
ENDIF ()

##########
# Installs
##########

INSTALL ( DIRECTORY
  ${OPENVDB_HOUDINI_UTILS_DIR}
  ${OPENVDB_HOUDINI_LOCAL_DIR}
  DESTINATION ${OPENVDB_HOUDINI_INSTALL_PREFIX}/include
)

INSTALL ( TARGETS
  openvdb_houdini
  DESTINATION ${OPENVDB_HOUDINI_INSTALL_PREFIX}/lib
)

FOREACH ( DSO_NAME ${OPENVDB_DSO_NAMES} )
  IF ( DSO_NAME MATCHES "VRAY_*")
    INSTALL ( TARGETS
      ${DSO_NAME}
      DESTINATION ${OPENVDB_HOUDINI_INSTALL_PREFIX}/dso/mantra
      )
  ELSE ()
    INSTALL ( TARGETS
      ${DSO_NAME}
      DESTINATION ${OPENVDB_HOUDINI_INSTALL_PREFIX}/dso
      )
  ENDIF ()
ENDFOREACH ()

INSTALL ( FILES
  houdini/SOP_OpenVDB.svg
  DESTINATION ${OPENVDB_HOUDINI_INSTALL_PREFIX}/config/Icons
  )

INSTALL ( FILES
  houdini/DW_OpenVDBRasterizePoints.cmd
  DESTINATION ${OPENVDB_HOUDINI_INSTALL_PREFIX}/scripts/sop
  )
