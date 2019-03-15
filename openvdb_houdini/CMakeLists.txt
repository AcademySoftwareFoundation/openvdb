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
  SHOPs, procedurals etc.) By default, both are installed to the same
  prefix:

    <prefix>/include  - Shared openvdb_houdini headers
    <prefix>/lib      - Location of libopenvdb_houdini
    <prefix>/dso      - Location of all other .so

  Where prefix is ${CMAKE_INSTALL_PREFIX}/houdini<version>. If you're
  building the Houdini plug-ins independently of the Core OpenVDB library,
  users may wish to set CMAKE_INSTALL_PREFIX to $HIH (provided by the
  Houdini Environment).

  If you wish to build and install the Houdini plug-in alongside the core
  library, or simply wish to customize the install paths of the above,
  the following variables can be set and provided to CMake:

   OPENVDB_HOUDINI_INSTALL_LIB_BASE_DIR
     - The prefix for the shared library and headers
   OPENVDB_HOUDINI_INSTALL_BASE_DIR
     - The prefix for all other dsos, icons, config files etcs.

#]=======================================================================]

PROJECT ( OpenVDBHoudini )
CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )
IF ( POLICY CMP0074 )
  CMAKE_POLICY ( SET CMP0074 NEW )
ENDIF ()

# CMake options for installation paths (if not overriding OPENVDB_HOUDINI_INSTALL variables)
OPTION ( OPENVDB_HOUDINI_SUBDIR "Install plugins to <prefix>/houdini/<version> instead of <prefix>/houdini<version>" OFF )
OPTION ( OPENVDB_HOUDINI_SHORT_VERSION "User major.minor as <version> instead of major.minor.patch" ON )
OPTION ( OPENVDB_HOUDINI_INSTALL_LIBRARY "Install headers and place the shared library in a lib directory." ON )

# SESITAG CMake options
OPTION ( OPENVDB_HOUDINI_CACHE_ONCE_SESI_TAG "Build the SESI Tag information on the first run of CMake only." ON )
OPTION ( OPENVDB_HOUDINI_DISABLE_SESI_TAG [=[
Disable SESI tag information on the OpenVDB DSOs. This is for CI/testing only - dsos without"
tag information will not load correctly.]=] OFF )

SET ( CUSTOM_SESI_TAG_INFO "CMake ${CMAKE_VERSION}"
  CACHE STRING "Custom data to feed to sesitag when compiling the Houdini DSOs." )

#########################################################################

MESSAGE ( STATUS "----------------------------------------------------" )
MESSAGE ( STATUS "------------ Configuring OpenVDBHoudini ------------" )
MESSAGE ( STATUS "----------------------------------------------------" )

#########################################################################

# Configure installation - most of the Houdini variables are set by the
# root CMake search for the Houdini Installation

SET ( HOUDINI_FOLDER_VERSION "${Houdini_VERSION_MAJOR}.${Houdini_VERSION_MINOR}" )
IF ( NOT OPENVDB_HOUDINI_SHORT_VERSION )
  SET ( HOUDINI_FOLDER_VERSION "${HOUDINI_FOLDER_VERSION}.${Houdini_VERSION_PATCH}" )
ENDIF ()

IF ( OPENVDB_HOUDINI_SUBDIR )
  SET ( HOUDINI_SUB_DIRECTORY "houdini/${HOUDINI_FOLDER_VERSION}" )
ELSE ()
  SET ( HOUDINI_SUB_DIRECTORY "houdini${HOUDINI_FOLDER_VERSION}" )
ENDIF ()

SET ( OPENVDB_HOUDINI_INSTALL_BASE_DIR
  ${CMAKE_INSTALL_PREFIX}/${HOUDINI_SUB_DIRECTORY}
  CACHE PATH "Base install path for OpenVDB Houdini nodes and icons."
  )
SET ( OPENVDB_HOUDINI_INSTALL_LIB_BASE_DIR
  ${OPENVDB_HOUDINI_INSTALL_BASE_DIR}
  CACHE PATH "Base install path for OpenVDB Houdini shared libraries and headers."
  )

MESSAGE ( STATUS "Houdini base DSO install path: ${OPENVDB_HOUDINI_INSTALL_BASE_DIR}" )
MESSAGE ( STATUS "Houdini base lib install path: ${OPENVDB_HOUDINI_INSTALL_LIB_BASE_DIR}" )

#########################################################################

# Collect and configure lib dependencies

IF ( NOT OPENVDB_BUILD_CORE )
  FIND_PACKAGE ( OpenVDB REQUIRED )
  SET ( OPENVDB_LIB OpenVDB::openvdb )
ELSE ()
  SET ( OPENVDB_LIB openvdb_shared )
  INCLUDE_DIRECTORIES ( ../ )
ENDIF ()

SET ( OPENVDB_HOUDINI_DEPENDANT_LIBS
  ${OPENVDB_LIB}
)
# Checking if hdk version is >= 17.5 Houdini 17.5 stopped shipping ilmbase
# and openexr so users can optionally set ILMBASE_/OPENEXR_ variables to find
# their own ILMBASE_/OPENEXR_ # installation
IF (( Houdini_VERSION_MAJOR GREATER 17 ) OR
    (( Houdini_VERSION_MAJOR EQUAL 17 ) AND ( Houdini_VERSION_MINOR EQUAL 5 )))
  FIND_PACKAGE ( IlmBase REQUIRED COMPONENTS Half )
  LIST ( APPEND OPENVDB_HOUDINI_DEPENDANT_LIBS IlmBase::Half )
ENDIF ()

# Checking if hdk version is >= 16.5 Houdini 16.5 stopped shipping boost.
# so users can optionally set BOOST_ variables to find their own BOOST_
# installation
IF (( Houdini_VERSION_MAJOR GREATER 16 ) OR
    (( Houdini_VERSION_MAJOR EQUAL 16 ) AND ( Houdini_VERSION_MINOR EQUAL 5 )))
  FIND_PACKAGE ( Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS iostreams system )
  LIST ( APPEND OPENVDB_HOUDINI_DEPENDANT_LIBS
    Boost::iostreams
    Boost::system
    )
ENDIF ()

IF ( USE_LOG4CPLUS )
  FIND_PACKAGE ( Log4cplus REQUIRED )
  LIST ( APPEND OPENVDB_HOUDINI_DEPENDANT_LIBS Log4cplus::log4cplus )
ENDIF ()

FIND_PACKAGE ( TBB REQUIRED COMPONENTS tbb )
LIST ( APPEND OPENVDB_HOUDINI_DEPENDANT_LIBS TBB::tbb )

IF ( USE_BLOSC )
  FIND_PACKAGE ( Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED )
  LIST ( APPEND OPENVDB_HOUDINI_DEPENDANT_LIBS Blosc::blosc )
ENDIF ()

# @note Linking openvdb_houdini against Houdini brings in all interface compile
# definitions, including MAKING_DSO
LIST ( APPEND OPENVDB_HOUDINI_DEPENDANT_LIBS Houdini )

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

INCLUDE_DIRECTORIES (
  ${OPENVDB_HOUDINI_UTILS_DIR}
  ${OPENVDB_HOUDINI_LOCAL_DIR}
  ${PROJECT_BINARY_DIR}
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

TARGET_LINK_LIBRARIES ( openvdb_houdini
  ${OPENVDB_HOUDINI_DEPENDANT_LIBS}
  )

IF ( OPENVDB_ENABLE_RPATH )
  # @todo There is probably a better way to do this for imported targets
  SET ( RPATHS )
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

# If OPENVDB_HOUDINI_CACHE_ONCE_SESI_TAG is ON, manually run the sesitag binary
# and cache the result for all subsequent runs.
# @note _houdini_bin_dir is set from the HoudiniConfig.cmake module

SET ( SESI_CACHE_TAG_DEFINE "" )
IF ( OPENVDB_HOUDINI_CACHE_ONCE_SESI_TAG AND NOT OPENVDB_HOUDINI_DISABLE_SESI_TAG )
  SET ( sesitag_input_file ${CMAKE_CURRENT_BINARY_DIR}/sesitag_input.cache/sesitag_input )

  IF ( NOT EXISTS ${sesitag_input_file} )
    STRING ( TIMESTAMP timestamp "%a %b %d %H:%M:%S %Y" )
    SET ( tag_info "Generated on: ${timestamp}\n${CUSTOM_SESI_TAG_INFO}" )
    FILE ( WRITE ${sesitag_input_file} "${tag_info}" )
  ENDIF ()

  SET ( sesitag_path "${_houdini_bin_dir}/sesitag${CMAKE_EXECUTABLE_SUFFIX}" )
  EXECUTE_PROCESS (
    COMMAND ${sesitag_path} -f ${sesitag_input_file}
    RESULT_VARIABLE status_code
    OUTPUT_VARIABLE SESI_CACHE_TAG_DEFINE
  )

  IF ( NOT status_code EQUAL 0 )
    MESSAGE ( FATAL_ERROR "Error running sesitag: ${status_code}" )
  ENDIF ()

  STRING ( REPLACE "'" "" SESI_CACHE_TAG_DEFINE ${SESI_CACHE_TAG_DEFINE} )
  STRING ( REPLACE "-D" "" SESI_CACHE_TAG_DEFINE ${SESI_CACHE_TAG_DEFINE} )
  MESSAGE ( STATUS "Using cache sesi tag data for Houdini DSOs: ${sesitag_input_file}" )
ENDIF ()

# Add all dsos as libraries and configure them for houdini

MESSAGE ( STATUS "Configuring Houdini DSOs..." )

FOREACH ( DSO_NAME ${OPENVDB_DSO_NAMES} )

  # Create the lib

  ADD_LIBRARY ( ${DSO_NAME} SHARED
    houdini/${DSO_NAME}.cc
    )

  # Link in deps

  TARGET_LINK_LIBRARIES ( ${DSO_NAME}
    openvdb_houdini
    ${OPENVDB_HOUDINI_DEPENDANT_LIBS}
    )

  # Call houdini_configure_target to setup the sesi tag information. We set
  # INSTDIR to the project build directory and configure the actuall install
  # paths ourselves (otherwise cmake will configure the build to build directly
  # to the install dir)

  houdini_configure_target ( ${DSO_NAME}
    INSTDIR ${PROJECT_BINARY_DIR}
    PREFIX ""
    TAGINFO ${CUSTOM_SESI_TAG_INFO}
  )

  # Add openvdb c flags

  IF ( USE_LOG4CPLUS )
    TARGET_COMPILE_DEFINITIONS ( ${DSO_NAME} PRIVATE "-DOPENVDB_USE_LOG4CPLUS" )
  ENDIF ()

  # If the sesi tag needs replacing or removing, manually configure the compile
  # options for each DSO and replace the UT_DSO_TAGINFO define
  # @todo ask for more options from sidefx for controlling tag timestamps

  IF ( OPENVDB_HOUDINI_DISABLE_SESI_TAG OR SESI_CACHE_TAG_DEFINE )
    GET_TARGET_PROPERTY ( DSO_DEFINITIONS ${DSO_NAME} COMPILE_DEFINITIONS )
    STRING ( REGEX REPLACE "UT_DSO_TAGINFO=\".+\"" "${SESI_CACHE_TAG_DEFINE}" DSO_DEFINITIONS "${DSO_DEFINITIONS}")
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
      SET_TARGET_PROPERTIES ( ${DSO_NAME}
        PROPERTIES INSTALL_RPATH "${RPATHS}"
        )
  ENDIF ()

  ADD_DEPENDENCIES ( openvdb_houdini_dsos ${DSO_NAME} )

ENDFOREACH ()

# This is only required for the OpenVDB Write SOP which enables a blosc menu
# option if blosc is supported.
# @todo If building against an external VDB lib, this should query if the lib
# was built with blosc
IF ( USE_BLOSC )
  TARGET_COMPILE_DEFINITIONS ( SOP_OpenVDB_Write PRIVATE "-DOPENVDB_USE_BLOSC" )
ENDIF ()

##########
# Installs
##########

IF ( OPENVDB_HOUDINI_INSTALL_LIBRARY )

  INSTALL ( DIRECTORY
    ${OPENVDB_HOUDINI_UTILS_DIR}
    ${OPENVDB_HOUDINI_LOCAL_DIR}
    DESTINATION ${OPENVDB_HOUDINI_INSTALL_LIB_BASE_DIR}/include
    )

  INSTALL ( TARGETS
    openvdb_houdini
    DESTINATION ${OPENVDB_HOUDINI_INSTALL_LIB_BASE_DIR}/lib
    )

ELSE ()

  INSTALL ( TARGETS
    openvdb_houdini
    DESTINATION ${OPENVDB_HOUDINI_INSTALL_LIB_BASE_DIR}/dso
    )

ENDIF ()

FOREACH ( DSO_NAME ${OPENVDB_DSO_NAMES} )
  INSTALL ( TARGETS
    ${DSO_NAME}
    DESTINATION ${OPENVDB_HOUDINI_INSTALL_BASE_DIR}/dso
    )
ENDFOREACH ()

INSTALL ( TARGETS
  VRAY_OpenVDB_Points
  DESTINATION ${OPENVDB_HOUDINI_INSTALL_BASE_DIR}/dso/mantra
  )

INSTALL ( FILES
  houdini/SOP_OpenVDB.svg
  DESTINATION ${OPENVDB_HOUDINI_INSTALL_BASE_DIR}/config/Icons
  )

INSTALL ( FILES
  houdini/DW_OpenVDBRasterizePoints.cmd
  DESTINATION ${OPENVDB_HOUDINI_INSTALL_BASE_DIR}/scripts/sop
  )
