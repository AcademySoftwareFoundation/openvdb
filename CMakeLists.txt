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

# yue.nicholas@gmail.com

CMAKE_MINIMUM_REQUIRED ( VERSION 3.1 )

SET ( CMAKE_CXX_STANDARD 11 )
SET ( CMAKE_DISABLE_SOURCE_CHANGES ON )
SET ( CMAKE_DISABLE_IN_SOURCE_BUILD ON )

PROJECT ( OpenVDB )

ENABLE_TESTING ()

FIND_PROGRAM ( CCACHE_FOUND ccache )
IF ( CCACHE_FOUND )
    SET_PROPERTY ( GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache )
    SET_PROPERTY ( GLOBAL PROPERTY RULE_LAUNCH_LINK ccache )
ENDIF ( CCACHE_FOUND )

OPTION ( OPENVDB_BUILD_UNITTESTS "Build the OpenVDB unit tests" ON )
OPTION ( OPENVDB_BUILD_DOCS "Build the OpenVDB documentation" OFF )
OPTION ( OPENVDB_BUILD_PYTHON_MODULE "Build the pyopenvdb Python module" OFF )
OPTION ( OPENVDB_BUILD_HOUDINI_SOPS "Build the Houdini SOPs" OFF )
OPTION ( OPENVDB_BUILD_MAYA_PLUGIN "Build the Maya plugin" OFF )
OPTION ( OPENVDB_BUILD_CORE "Build the OpenVDB core" ON )
OPTION ( OPENVDB_ENABLE_RPATH "Build with RPATH information" ON )
OPTION ( OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING "Disable the implicit linking of Boost libraries on Windows" ON )
OPTION ( OPENVDB_CXX_STRICT "Enable or disable a set of pre-defined compiler warnings for clang and gcc" OFF )
OPTION ( OPENVDB_ENABLE_3_ABI_COMPATIBLE "Build with OpenVDB ABI 3 (deprecated - see "
  "OPENVDB_ABI_VERSION_NUMBER)" OFF )

SET ( OPENVDB_ABI_VERSION_NUMBER "" CACHE STRING "Build for compatibility with version N of the
    OpenVDB Grid ABI, where N is 3, 4, 5 etc. (some newer features will be disabled)")

IF ( OPENVDB_BUILD_HOUDINI_SOPS )
  # @todo why?
  SET ( OPENVDB_BUILD_PYTHON_MODULE OFF )
ENDIF ()

# Some platform checks

IF ( APPLE )
  IF ( OPENVDB_BUILD_HOUDINI_SOPS AND OPENVDB_BUILD_MAYA_PLUGIN )
  MESSAGE ( FATAL_ERROR "On OSX, Houdini linked with libc++ whereas Maya is linked with libstdc++, "
        "the different runtimes are not mutually compatible, build them Houdini SOPs and Maya plugins "
        "separately with the requisite DCC's compiler" )
  ENDIF ()
ENDIF ()

# Determine OpenVDB version

SET ( OPENVDB_VERSION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/version.h )
FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_major_version_str REGEX "^#define[\t ]+OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+.*")
FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_minor_version_str REGEX "^#define[\t ]+OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+.*")
FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_patch_version_str REGEX "^#define[\t ]+OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+.*")

STRING (REGEX REPLACE "^.*OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
  _openvdb_major_version_number "${openvdb_major_version_str}")
STRING (REGEX REPLACE "^.*OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
  _openvdb_minor_version_number "${openvdb_minor_version_str}")
STRING (REGEX REPLACE "^.*OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
  _openvdb_patch_version_number "${openvdb_patch_version_str}")

SET ( OPENVDB_MAJOR_VERSION_NUMBER ${_openvdb_major_version_number} CACHE STRING "OpenVDB Major Version Number" )
SET ( OPENVDB_MINOR_VERSION_NUMBER ${_openvdb_minor_version_number} CACHE STRING "OpenVDB Minor Version Number" )
SET ( OPENVDB_PATCH_VERSION_NUMBER ${_openvdb_patch_version_number} CACHE STRING "OpenVDB Build Version Number" )

# Determine the OpenVDB ABI Version. If OPENVDB_ABI_VERSION_NUMBER is defined, use that value,
# otherwise check the legacy OPENVDB_ENABLE_3_ABI_COMPATIBLE option, falling back to the major
# library version number
MESSAGE ( STATUS "Using OpenVDB ABI Version ${OPENVDB_ABI_VERSION_NUMBER}" )

IF ( OPENVDB_ABI_VERSION_NUMBER STREQUAL "" )
  IF ( OPENVDB_ENABLE_3_ABI_COMPATIBLE )
    SET ( OPENVDB_ABI_VERSION_NUMBER 3 )
    ADD_DEFINITIONS ( -DOPENVDB_3_ABI_COMPATIBLE )
    MESSAGE ( DEPRECATION "OPENVDB_ENABLE_3_ABI_COMPATIBLE is deprecated. Instead use
      -D OPENVDB_ABI_VERSION_NUMBER=N, where N is the abi version." )
  ELSE ()
    SET ( OPENVDB_ABI_VERSION_NUMBER ${OPENVDB_MAJOR_VERSION_NUMBER} )
  ENDIF ()
ENDIF ()

MESSAGE ( STATUS "Using OpenVDB ABI Version ${OPENVDB_ABI_VERSION_NUMBER}" )
ADD_DEFINITIONS ( -DOPENVDB_ABI_VERSION_NUMBER=${OPENVDB_ABI_VERSION_NUMBER} )

# Configure the top level dir to make sure we pick up headers from this version of OpenVDB we
# are building

LIST ( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )
SET ( OPENVDB_TOP_LEVEL_DIR ${PROJECT_SOURCE_DIR} CACHE PATH
    "OpenVDB Top Level Directory to share among sub projects" )
INCLUDE_DIRECTORIES ( ${OPENVDB_TOP_LEVEL_DIR} )

# Compiler options

IF ( OPENVDB_CXX_STRICT )
    # Add definitions for a number of compiler warnings for GCC and Clang for all sub projects
    IF ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
        MESSAGE ( STATUS "Configuring Clang CXX warnings" )
        ADD_DEFINITIONS (
            -Wall
            -Wextra
            -Wconversion
            -Wno-sign-conversion
        )
    ELSEIF ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
        MESSAGE ( STATUS "Configuring GCC CXX warnings" )
        ADD_DEFINITIONS (
            -Wall
            -Wextra
            -pedantic
            -Wcast-align
            -Wcast-qual
            -Wconversion
            -Wdisabled-optimization
            -Woverloaded-virtual
        )
    ELSE ()
        MESSAGE ( WARNING "No available CXX warnings for compiler ${CMAKE_CXX_COMPILER_ID}" )
    ENDIF ()
ENDIF ()

IF ( OPENVDB_BUILD_CORE )
  ADD_SUBDIRECTORY ( openvdb )
ENDIF ()

IF (OPENVDB_BUILD_PYTHON_MODULE)
  ADD_SUBDIRECTORY ( openvdb/python )
ENDIF ()

IF ( OPENVDB_BUILD_HOUDINI_SOPS )
  ADD_SUBDIRECTORY ( openvdb_houdini )
ENDIF ()

IF ( OPENVDB_BUILD_MAYA_PLUGIN )
  ADD_SUBDIRECTORY ( openvdb_maya )
ENDIF ()
