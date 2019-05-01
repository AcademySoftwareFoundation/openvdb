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

OpenVDBMayaSetup
-------------------

Finds the Maya installation and sets up dependencies for OpenVDB builds.
This ensures that all dependencies that are included with a Maya
distribution are configured to load from that installation.

Use this module by invoking include with the form::

  include ( OpenVDBMayaSetup )

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Maya_FOUND``
  True if the system has Maya installed.
``Maya_VERSION``
  The version of the Maya which was found.

Additionally, the following values are set for all dependent OpenVDB
builds, ensuring they link against the correct libraries. This may
overwrite user provided values.

``TBB_INCLUDEDIR``
``TBB_LIBRARYDIR``

Hints
^^^^^

Instead of explicitly setting the cache variables, the following
variables may be provided to tell this module where to look.

``MAYA_ROOT``
  Preferred installation prefix.
``ENV{MAYA_ROOT}``
  Preferred installation prefix.
``ENV{MAYA_LOCATION}``
  Preferred installation prefix.

#]=======================================================================]

# Find the Maya installation and use Maya's CMake to initialize
# the Maya lib

SET ( _MAYA_ROOT_SEARCH_DIR )

IF ( MAYA_ROOT )
  LIST ( APPEND _MAYA_ROOT_SEARCH_DIR ${MAYA_ROOT} )
ELSE ()
  SET ( _ENV_MAYA_ROOT $ENV{MAYA_ROOT} )
  IF ( _ENV_MAYA_ROOT )
    LIST ( APPEND _MAYA_ROOT_SEARCH_DIR ${_ENV_MAYA_ROOT} )
  ENDIF ()
  SET ( _ENV_MAYA_ROOT $ENV{MAYA_LOCATION} )
  IF ( _ENV_MAYA_ROOT )
    LIST ( APPEND _MAYA_ROOT_SEARCH_DIR ${_ENV_MAYA_ROOT} )
  ENDIF ()
ENDIF ()

# ------------------------------------------------------------------------
#  Search for Maya
# ------------------------------------------------------------------------

FIND_PATH ( Maya_INCLUDE_DIR maya/MTypes.h
  NO_DEFAULT_PATH
  PATHS ${_MAYA_ROOT_SEARCH_DIR}
  PATH_SUFFIXES include
  )

IF ( NOT EXISTS "${Maya_INCLUDE_DIR}/maya/MTypes.h" )
  MESSAGE ( FATAL_ERROR "Unable to locate Maya Installation." )
ENDIF ()

# Determine Maya version, including point releases. Currently only works for
# Maya 2016 and onwards so there is no -x64 and -x32 suffixes in the version
FILE ( STRINGS "${Maya_INCLUDE_DIR}/maya/MTypes.h"
  _maya_version_string REGEX "#define MAYA_API_VERSION "
  )
STRING ( REGEX REPLACE ".*#define[ \t]+MAYA_API_VERSION[ \t]+([0-9]+).*$" "\\1"
  _maya_version_string "${_maya_version_string}"
  )
STRING ( SUBSTRING ${_maya_version_string} 0 4 Maya_MAJOR_VERSION )
STRING ( SUBSTRING ${_maya_version_string} 4 2 Maya_MINOR_VERSION )

IF ( Maya_MINOR_VERSION LESS 50 )
  SET ( Maya_VERSION ${Maya_MAJOR_VERSION} )
ELSE ()
  SET ( Maya_VERSION ${Maya_MAJOR_VERSION}.5 )
ENDIF ()
UNSET ( _maya_version_string )

# Find required maya libs

SET ( _MAYA_COMPONENT_LIST
  OpenMaya
  OpenMayaFX
  OpenMayaUI
  Foundation
  )

SET ( Maya_LIBRARY_DIR "" )
IF ( APPLE )
  SET ( Maya_LIBRARY_DIR ${Maya_INCLUDE_DIR}/../Maya.app/Contents/MacOS/ )
ELSE ()
  SET ( Maya_LIBRARY_DIR ${Maya_INCLUDE_DIR}/../lib/ )
ENDIF ()

SET ( Maya_LIB_COMPONENTS "" )

FOREACH ( COMPONENT ${_MAYA_COMPONENT_LIST} )
  FIND_LIBRARY ( Maya_${COMPONENT}_LIBRARY ${COMPONENT}
    NO_DEFAULT_PATH
    PATHS ${Maya_LIBRARY_DIR}
    )
  LIST ( APPEND Maya_LIB_COMPONENTS ${Maya_${COMPONENT}_LIBRARY} )
ENDFOREACH ()

# ------------------------------------------------------------------------
#  Cache and set Maya_FOUND
# ------------------------------------------------------------------------

INCLUDE ( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( Maya
  FOUND_VAR Maya_FOUND
  REQUIRED_VARS
    Maya_INCLUDE_DIR
    Maya_LIB_COMPONENTS
  VERSION_VAR Maya_VERSION
)

IF ( NOT Maya_FOUND )
  MESSAGE ( FATAL_ERROR "Unable to locate Maya Installation." )
ELSEIF ( Maya_VERSION VERSION_LESS MINIMUM_MAYA_VERSION )
  MESSAGE ( WARNING "Unsupported Maya Version ${Maya_VERSION}. Minimum "
    "supported is ${MINIMUM_MAYA_VERSION}."
    )
ENDIF ()

# ------------------------------------------------------------------------
#  Configure dependencies
# ------------------------------------------------------------------------

IF ( NOT TBB_INCLUDEDIR )
  SET ( TBB_INCLUDEDIR ${Maya_INCLUDE_DIR} )
ENDIF ()
IF ( NOT TBB_LIBRARYDIR )
  SET ( TBB_LIBRARYDIR ${Maya_LIBRARY_DIR} )
ENDIF ()

# ------------------------------------------------------------------------
#  Configure Maya
# ------------------------------------------------------------------------

SET ( Maya_LIBRARIES ${Maya_LIB_COMPONENTS} )
SET ( Maya_INCLUDE_DIRS ${Maya_INCLUDE_DIR} )
SET ( Maya_LIBRARY_DIRS ${Maya_LIBRARY_DIR} )

IF ( APPLE )
  SET ( Maya_DEFINITIONS
    -DMAC_PLUGIN
    -DREQUIRE_IOSTREAM
    -DOSMac_
    -DOSMac_MachO_
    -D_BOOL
    )
ELSEIF ( WIN32 )
  SET ( Maya_DEFINITIONS
    -DNOMINMAX
    -DNT_PLUGIN
    -DREQUIRE_IOSTREAM
    -D_USE_MATH_DEFINES
    -D_CRT_SECURE_NO_WARNINGS
    )
ELSE ()
  SET ( Maya_DEFINITIONS
    -D_BOOL
    -DFUNCPROTO
    -DGL_GLEXT_PROTOTYPES=1
    -DREQUIRE_IOSTREAM
    -DUNIX
    -fno-gnu-keywords
    -fno-omit-frame-pointer
    -fno-strict-aliasing
    -funsigned-char
    -Wno-comment
    -Wno-multichar
    -Wno-strict-aliasing
    -m64
    -DBits64_
    -DLINUX
    -DLINUX_64
    )
ENDIF ()

# Configure imported targets

IF ( NOT TARGET Maya )
  ADD_LIBRARY ( Maya INTERFACE )
  FOREACH ( COMPONENT ${_MAYA_COMPONENT_LIST} )
    ADD_LIBRARY ( Maya::${COMPONENT} UNKNOWN IMPORTED )
    SET_TARGET_PROPERTIES ( Maya::${COMPONENT} PROPERTIES
      IMPORTED_LOCATION "${Maya_${COMPONENT}_LIBRARY}"
      INTERFACE_COMPILE_OPTIONS "${Maya_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${Maya_INCLUDE_DIRS}"
    )
    TARGET_LINK_LIBRARIES ( Maya INTERFACE Maya::${COMPONENT} )
  ENDFOREACH ()
ENDIF ()

MACRO( MAYA_SET_LIBRARY_PROPERTIES NAME )
  IF (WIN32)
    SET_TARGET_PROPERTIES ( ${NAME} PROPERTIES
      SUFFIX ".mll"
      PREFIX ""
      LINK_FLAGS "/export:initializePlugin /export:uninitializePlugin"
      )
  ELSEIF (APPLE)
    SET_TARGET_PROPERTIES ( ${NAME} PROPERTIES
      SUFFIX ".bundle"
      PREFIX "")
  ELSE ()
    SET_TARGET_PROPERTIES ( ${NAME} PROPERTIES
      PREFIX "")
  ENDIF ()
ENDMACRO ()
