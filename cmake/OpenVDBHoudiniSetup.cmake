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
#[=======================================================================[.rst:

OpenVDBHoudiniSetup
-------------------

Wraps the call the FindPackage ( Houdini ) for OpenVDB builds. This
ensures that all dependencies that are included with a Houdini
distribution are configured to load from that installation.

This CMake searches for the HoudiniConfig.cmake module provided by
SideFX to configure the OpenVDB Houdini base and DSO libraries. Users
can provide paths to the location of their Houdini Installation by
setting HOUDINI_ROOT either as an environment variable or by passing it
to CMake. This module also reads the value of $HFS, usually set by
sourcing the Houdini Environment. Note that as long as you provide a
path to your Houdini Installation you do not need to source the
Houdini Environment.

Use this module by invoking include with the form::

  include ( OpenVDBHoudiniSetup )

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Houdini_FOUND``
  True if the system has Houdini installed.
``Houdini_VERSION``
  The version of the Houdini which was found.
``OPENVDB_HOUDINI_ABI``
  The ABI version that Houdini uses for it's own OpenVDB installation.

A variety of variables will also be set from HoudiniConfig.cmake.

Additionally, the following values are set for all dependent OpenVDB
builds, ensuring they link against the correct libraries. This may
overwrite user provided values.

``ZLIB_ROOT``
``ZLIB_LIBRARY``
``OPENEXR_INCLUDEDIR``
``OPENEXR_LIBRARYDIR``
``ILMBASE_INCLUDEDIR``
``ILMBASE_LIBRARYDIR``
``TBB_INCLUDEDIR``
``TBB_LIBRARYDIR``
``BLOSC_INCLUDEDIR``
``BLOSC_LIBRARYDIR``

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``ENV{HFS}``
  Preferred installation prefix.
``HOUDINI_ROOT``
  Preferred installation prefix.
``CMAKE_PREFIX_PATH``
  Add the location of your Houdini installations CMake to this path.

#]=======================================================================]

# Find the Houdini installation and use Houdini's CMake to initialize
# the Houdini lib

SET ( _HOUDINI_ROOT_SEARCH_DIR )

IF ( HOUDINI_ROOT )
  LIST ( APPEND _HOUDINI_ROOT_SEARCH_DIR ${HOUDINI_ROOT} )
ELSE ()
  SET ( _ENV_HOUDINI_ROOT $ENV{HOUDINI_ROOT} )
  IF ( _ENV_HOUDINI_ROOT )
    LIST ( APPEND _HOUDINI_ROOT_SEARCH_DIR ${_ENV_HOUDINI_ROOT} )
  ENDIF ()
ENDIF ()

IF ( DEFINED ENV{HFS} )
  LIST ( APPEND _HOUDINI_ROOT_SEARCH_DIR $ENV{HFS} )
ENDIF ()

# ------------------------------------------------------------------------
#  Search for Houdini CMake
# ------------------------------------------------------------------------

SET ( _HOUDINI_CMAKE_PATH_SUFFIXES )

IF ( APPLE )
  LIST ( APPEND _HOUDINI_CMAKE_PATH_SUFFIXES
    Frameworks/Houdini.framework/Versions/Current/Resources/toolkit/cmake
    Houdini.framework/Versions/Current/Resources/toolkit/cmake
    Versions/Current/Resources/toolkit/cmake
    Current/Resources/toolkit/cmake
    Resources/toolkit/cmake
    )
ENDIF ()

LIST ( APPEND _HOUDINI_CMAKE_PATH_SUFFIXES
  toolkit/cmake
  cmake
  )

FIND_PATH ( HOUDINI_CMAKE_LOCATION HoudiniConfig.cmake
  NO_DEFAULT_PATH
  PATHS ${_HOUDINI_ROOT_SEARCH_DIR}
  PATH_SUFFIXES ${_HOUDINI_CMAKE_PATH_SUFFIXES}
  )

IF ( HOUDINI_CMAKE_LOCATION )
  LIST ( APPEND CMAKE_PREFIX_PATH "${HOUDINI_CMAKE_LOCATION}" )
ENDIF ()

FIND_PACKAGE ( Houdini REQUIRED )

# Note that passing MINIMUM_HOUDINI_VERSION into FIND_PACKAGE ( Houdini ) doesn't work
IF ( NOT Houdini_FOUND )
  MESSAGE ( FATAL_ERROR "Unable to locate Houdini Installation." )
ELSEIF ( Houdini_VERSION VERSION_LESS MINIMUM_HOUDINI_VERSION )
  MESSAGE ( FATAL_ERROR "Unsupported Houdini Version ${Houdini_VERSION}. Minimum "
    "supported is ${MINIMUM_HOUDINI_VERSION}."
    )
ENDIF ()

FIND_PACKAGE ( PackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( Houdini
  REQUIRED_VARS _houdini_install_root Houdini_FOUND
  VERSION_VAR Houdini_VERSION
  )

# ------------------------------------------------------------------------
#  Add support for older versions of Houdini
# ------------------------------------------------------------------------

IF ( Houdini_VERSION VERSION_LESS 17 )
  # Missing function in Houdini 16.5 CMake copied from 17.5 - _houdini variables
  # are set by the Houdini configuration package
  function ( houdini_get_default_install_dir output_var )
    set( _instdir "" )
    if ( _houdini_platform_linux )
        set( _instdir $ENV{HOME}/houdini${_houdini_release_version} )
    elseif ( _houdini_platform_osx )
        set( _instdir $ENV{HOME}/Library/Preferences/houdini/${_houdini_release_version} )
    elseif ( _houdini_platform_win )
        set( _instdir $ENV{HOMEDRIVE}$ENV{HOMEPATH}\\Documents\\houdini${_houdini_release_version} )
    else ()
        message( FATAL_ERROR "Invalid platform" )
    endif ()
    set( ${output_var} ${_instdir} PARENT_SCOPE )
  endfunction ()
ENDIF ()

# ------------------------------------------------------------------------
#  Configure imported Houdini target
# ------------------------------------------------------------------------

# Set the relative directory containing Houdini libs and populate an extra list
# of Houdini dependencies for _houdini_create_libraries.

SET ( _HOUDINI_LIB_DIR )
SET ( _HOUDINI_EXTRA_LIBRARIES )
SET ( _HOUDINI_EXTRA_LIBRARY_NAMES )

IF ( APPLE )
  SET ( _HOUDINI_LIB_DIR
    Frameworks/Houdini.framework/Versions/Current/Libraries
    )
  LIST ( APPEND _HOUDINI_EXTRA_LIBRARIES
    ${_HOUDINI_LIB_DIR}/libHoudiniRAY.dylib
    ${_HOUDINI_LIB_DIR}/libhboost_regex.dylib
    )
ELSE ()
  SET ( _HOUDINI_LIB_DIR dsolib )
  LIST ( APPEND _HOUDINI_EXTRA_LIBRARIES
    ${_HOUDINI_LIB_DIR}/libHoudiniRAY.so
    ${_HOUDINI_LIB_DIR}/libhboost_regex.so
    )
ENDIF ()

LIST ( APPEND _HOUDINI_EXTRA_LIBRARY_NAMES
  HoudiniRAY
  hboost_regex
  )

# Additionally link extra deps

_houdini_create_libraries (
  PATHS ${_HOUDINI_EXTRA_LIBRARIES}
  TARGET_NAMES ${_HOUDINI_EXTRA_LIBRARY_NAMES}
  TYPE SHARED
  )

UNSET ( _HOUDINI_EXTRA_LIBRARIES )
UNSET ( _HOUDINI_EXTRA_LIBRARY_NAMES )

# Set Houdini lib and include directories

SET ( _HOUDINI_INCLUDE_DIR ${_houdini_include_dir} )
SET ( _HOUDINI_LIB_DIR ${_houdini_install_root}/${_HOUDINI_LIB_DIR} )

# ------------------------------------------------------------------------
#  Configure dependencies
# ------------------------------------------------------------------------

# Congfigure dependency hints to point to Houdini. Allow for user overriding
# if custom Houdini installations are in use

# ZLIB - FindPackage ( ZLIB ) only supports a few path hints. We use
# ZLIB_ROOT to find the zlib includes and explicitly set the path to
# the zlib library

IF ( NOT ZLIB_ROOT )
  SET ( ZLIB_ROOT ${_HOUDINI_INCLUDE_DIR} )
ENDIF ()
IF ( NOT ZLIB_LIBRARY )
  # Full path to zlib library - FindPackage ( ZLIB )
  FIND_LIBRARY ( ZLIB_LIBRARY z
    NO_DEFAULT_PATH
    PATHS ${_HOUDINI_LIB_DIR}
    )
  IF ( NOT EXISTS ${ZLIB_LIBRARY} )
    MESSAGE ( WARNING "The OpenVDB Houdini CMake setup is unable to locate libz within "
      "the Houdini installation at: ${_HOUDINI_LIB_DIR}. OpenVDB may not build correctly."
      )
  ENDIF ()
ENDIF ()

# TBB

IF ( NOT TBB_INCLUDEDIR )
  SET ( TBB_INCLUDEDIR ${_HOUDINI_INCLUDE_DIR} )
ENDIF ()
IF ( NOT TBB_LIBRARYDIR )
  SET ( TBB_LIBRARYDIR ${_HOUDINI_LIB_DIR} )
ENDIF ()

# Blosc

IF ( NOT BLOSC_INCLUDEDIR )
  SET ( BLOSC_INCLUDEDIR ${_HOUDINI_INCLUDE_DIR} )
ENDIF ()
IF ( NOT BLOSC_LIBRARYDIR )
  SET ( BLOSC_LIBRARYDIR ${_HOUDINI_LIB_DIR} )
ENDIF ()

# OpenEXR

IF ( NOT OPENEXR_INCLUDEDIR )
  SET ( OPENEXR_INCLUDEDIR ${_HOUDINI_INCLUDE_DIR} )
ENDIF ()
IF ( NOT OPENEXR_LIBRARYDIR )
  SET ( OPENEXR_LIBRARYDIR ${_HOUDINI_LIB_DIR} )
ENDIF ()

# IlmBase

IF ( NOT ILMBASE_INCLUDEDIR )
  SET ( ILMBASE_INCLUDEDIR ${_HOUDINI_INCLUDE_DIR} )
ENDIF ()
IF ( NOT ILMBASE_LIBRARYDIR )
  SET ( ILMBASE_LIBRARYDIR ${_HOUDINI_LIB_DIR} )
ENDIF ()

# Boost

IF ( Houdini_VERSION VERSION_LESS 16.5 )
  IF ( OPENVDB_BUILD_PYTHON_MODULE )
    # Prior to the introduction of HBoost (Houdini's namespaced and shipped Boost version from 16.5),
    # we built against Houdini's version of Boost which didn't include Boost.Python
    SET ( OPENVDB_BUILD_PYTHON_MODULE OFF )
    MESSAGE ( WARNING "Disabling compilation of the OpenVDB Python module. The python module requires "
      "Boost.Python which cannot be linked in when building against Houdini Version 16.0 and earlier." )
  ENDIF ()
  # Reset boost hints if not set
  IF ( NOT BOOST_INCLUDEDIR )
    SET ( BOOST_INCLUDEDIR ${_HOUDINI_INCLUDE_DIR} )
  ENDIF ()
  IF ( NOT BOOST_LIBRARYDIR )
    SET ( BOOST_LIBRARYDIR ${_HOUDINI_LIB_DIR} )
  ENDIF ()
ENDIF ()

UNSET ( _HOUDINI_INCLUDE_DIR )
UNSET ( _HOUDINI_LIB_DIR )

# Versions of Houdini >= 17.5 have some namespaced libraries (IlmBase/OpenEXR).
# Add the required suffix as part of the cmake lib suffix searches

IF ( APPLE )
  LIST ( APPEND CMAKE_FIND_LIBRARY_SUFFIXES "_sidefx.dylib" )
ELSEIF ( UNIX )
  LIST ( APPEND CMAKE_FIND_LIBRARY_SUFFIXES "_sidefx.so" )
ENDIF ()

# ------------------------------------------------------------------------
#  Configure OpenVDB ABI
# ------------------------------------------------------------------------

# Explicitly configure the OpenVDB ABI version depending on the Houdini
# version.

IF ( Houdini_VERSION VERSION_LESS 16.5 )
  SET ( OPENVDB_HOUDINI_ABI 3 )
ELSEIF ( Houdini_VERSION VERSION_LESS 17 )
  SET ( OPENVDB_HOUDINI_ABI 4 )
ELSE ()
  SET ( OPENVDB_HOUDINI_ABI 5 )
ENDIF ()
