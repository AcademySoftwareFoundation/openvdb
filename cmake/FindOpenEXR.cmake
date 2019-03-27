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

FindOpenEXR
-----------

Find OpenEXR include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(OpenEXR
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if OpenEXR is not found
    [COMPONENTS <libs>...] # OpenEXR libraries by their canonical name
                           # e.g. "IlmImf" for "libIlmImf"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``OpenEXR::IlmImf``
  The IlmImf library target.
``OpenEXR::IlmImfUtil``
  The IlmImfUtil library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``OpenEXR_FOUND``
  True if the system has the OpenEXR library.
``OpenEXR_VERSION``
  The version of the OpenEXR library which was found.
``OpenEXR_INCLUDE_DIRS``
  Include directories needed to use OpenEXR.
``OpenEXR_LIBRARIES``
  Libraries needed to link to OpenEXR.
``OpenEXR_LIBRARY_DIRS``
  OpenEXR library directories.
``OpenEXR_DEFINITIONS``
  Definitions to use when compiling code that uses OpenEXR.
``OpenEXR_{COMPONENT}_FOUND``
  True if the system has the named OpenEXR component.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OpenEXR_INCLUDE_DIR``
  The directory containing ``OpenEXR/config-auto.h``.
``OpenEXR_{COMPONENT}_LIBRARY``
  Individual component libraries for OpenEXR
``OpenEXR_{COMPONENT}_DLL``
  Individual component dlls for OpenEXR on Windows.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``OPENEXR_ROOT``
  Preferred installation prefix.
``OPENEXR_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``OPENEXR_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

MARK_AS_ADVANCED (
  OpenEXR_INCLUDE_DIR
  OpenEXR_LIBRARY
  OPENEXR_NAMESPACE_VERSIONING
)

SET ( _OPENEXR_COMPONENT_LIST
  IlmImf
  IlmImfUtil
  )

IF ( OpenEXR_FIND_COMPONENTS )
  SET ( OPENEXR_COMPONENTS_PROVIDED TRUE )
  SET ( _IGNORED_COMPONENTS "" )
  FOREACH ( COMPONENT ${OpenEXR_FIND_COMPONENTS} )
    IF ( NOT ${COMPONENT} IN_LIST _OPENEXR_COMPONENT_LIST )
      LIST ( APPEND _IGNORED_COMPONENTS ${COMPONENT} )
    ENDIF ()
  ENDFOREACH()

  IF ( _IGNORED_COMPONENTS )
    MESSAGE ( STATUS "Ignoring unknown components of OpenEXR:" )
    FOREACH ( COMPONENT ${_IGNORED_COMPONENTS} )
      MESSAGE ( STATUS "  ${COMPONENT}" )
    ENDFOREACH ()
    LIST ( REMOVE_ITEM OpenEXR_FIND_COMPONENTS ${_IGNORED_COMPONENTS} )
  ENDIF ()
ELSE ()
  SET ( OPENEXR_COMPONENTS_PROVIDED FALSE )
  SET ( OpenEXR_FIND_COMPONENTS ${_OPENEXR_COMPONENT_LIST} )
ENDIF ()

# Append OPENEXR_ROOT or $ENV{OPENEXR_ROOT} if set (prioritize the direct cmake var)
SET ( _OPENEXR_ROOT_SEARCH_DIR "" )

IF ( OPENEXR_ROOT )
  LIST ( APPEND _OPENEXR_ROOT_SEARCH_DIR ${OPENEXR_ROOT} )
ELSE ()
  SET ( _ENV_OPENEXR_ROOT $ENV{OPENEXR_ROOT} )
  IF ( _ENV_OPENEXR_ROOT )
    LIST ( APPEND _OPENEXR_ROOT_SEARCH_DIR ${_ENV_OPENEXR_ROOT} )
  ENDIF ()
ENDIF ()

# Additionally try and use pkconfig to find OpenEXR

FIND_PACKAGE ( PkgConfig )
PKG_CHECK_MODULES ( PC_OpenEXR QUIET OpenEXR )

# ------------------------------------------------------------------------
#  Search for OpenEXR include DIR
# ------------------------------------------------------------------------

SET ( _OPENEXR_INCLUDE_SEARCH_DIRS "" )
LIST ( APPEND _OPENEXR_INCLUDE_SEARCH_DIRS
  ${OPENEXR_INCLUDEDIR}
  ${_OPENEXR_ROOT_SEARCH_DIR}
  ${PC_OpenEXR_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Look for a standard OpenEXR header file.
FIND_PATH ( OpenEXR_INCLUDE_DIR OpenEXRConfig.h
  NO_DEFAULT_PATH
  PATHS ${_OPENEXR_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES  include/OpenEXR OpenEXR
  )

IF ( EXISTS "${OpenEXR_INCLUDE_DIR}/OpenEXRConfig.h" )
  # Get the EXR version information from the config header
  FILE ( STRINGS "${OpenEXR_INCLUDE_DIR}/OpenEXRConfig.h"
    _openexr_version_major_string REGEX "#define OPENEXR_VERSION_MAJOR "
    )
  STRING ( REGEX REPLACE "#define OPENEXR_VERSION_MAJOR" ""
    _openexr_version_major_string "${_openexr_version_major_string}"
    )
  STRING ( STRIP "${_openexr_version_major_string}" OpenEXR_VERSION_MAJOR )

  FILE ( STRINGS "${OpenEXR_INCLUDE_DIR}/OpenEXRConfig.h"
     _openexr_version_minor_string REGEX "#define OPENEXR_VERSION_MINOR "
    )
  STRING ( REGEX REPLACE "#define OPENEXR_VERSION_MINOR" ""
    _openexr_version_minor_string "${_openexr_version_minor_string}"
    )
  STRING ( STRIP "${_openexr_version_minor_string}" OpenEXR_VERSION_MINOR )

  UNSET ( _openexr_version_major_string )
  UNSET ( _openexr_version_minor_string )

  SET ( OpenEXR_VERSION ${OpenEXR_VERSION_MAJOR}.${OpenEXR_VERSION_MINOR} )
ENDIF ()

# ------------------------------------------------------------------------
#  Search for OPENEXR lib DIR
# ------------------------------------------------------------------------

SET ( _OPENEXR_LIBRARYDIR_SEARCH_DIRS "" )

# Append to _OPENEXR_LIBRARYDIR_SEARCH_DIRS in priority order

LIST ( APPEND _OPENEXR_LIBRARYDIR_SEARCH_DIRS
  ${OPENEXR_LIBRARYDIR}
  ${_OPENEXR_ROOT_SEARCH_DIR}
  ${PC_OpenEXR_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Build suffix directories

SET ( OPENEXR_PATH_SUFFIXES
  lib64
  lib
)

IF ( UNIX )
  LIST ( INSERT OPENEXR_PATH_SUFFIXES 0 lib/x86_64-linux-gnu )
ENDIF ()

SET ( _OPENEXR_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( OpenEXR_LIB_COMPONENTS "" )

FOREACH ( COMPONENT ${OpenEXR_FIND_COMPONENTS} )
  # library suffix handling
  IF ( WIN32 )
    SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".lib" )
  ENDIF ()

  IF ( OPENEXR_USE_STATIC_LIBS )
    IF ( UNIX )
      SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".a" )
    ENDIF ()
  ENDIF ()

  SET ( LIB_NAME ${COMPONENT} )
  IF ( OPENEXR_NAMESPACE_VERSIONING AND OpenEXR_VERSION )
    SET ( LIB_NAME "${LIB_NAME}-${OpenEXR_VERSION_MAJOR}_${OpenEXR_VERSION_MINOR}" )
  ENDIF ()

  FIND_LIBRARY ( OpenEXR_${COMPONENT}_LIBRARY ${LIB_NAME}
    NO_DEFAULT_PATH
    PATHS ${_OPENEXR_LIBRARYDIR_SEARCH_DIRS}
    PATH_SUFFIXES ${OPENEXR_PATH_SUFFIXES}
    )
  LIST ( APPEND OpenEXR_LIB_COMPONENTS ${OpenEXR_${COMPONENT}_LIBRARY} )

  IF ( NOT OPENEXR_USE_STATIC_LIBS AND WIN32 )
    SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".dll" )
    FIND_LIBRARY ( OpenEXR_${COMPONENT}_DLL ${LIB_NAME}
      NO_DEFAULT_PATH
      PATHS ${_OPENEXR_LIBRARYDIR_SEARCH_DIRS}
      PATH_SUFFIXES bin
      )
  ENDIF ()

  IF ( OpenEXR_${COMPONENT}_LIBRARY )
    SET ( OpenEXR_${COMPONENT}_FOUND TRUE )
  ELSE ()
    SET ( OpenEXR_${COMPONENT}_FOUND FALSE )
  ENDIF ()
ENDFOREACH ()

# reset lib suffix

SET ( CMAKE_FIND_LIBRARY_SUFFIXES ${_OPENEXR_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES} )

# ------------------------------------------------------------------------
#  Cache and set OPENEXR_FOUND
# ------------------------------------------------------------------------

INCLUDE ( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( OpenEXR
  FOUND_VAR OpenEXR_FOUND
  REQUIRED_VARS
    OpenEXR_INCLUDE_DIR
    OpenEXR_LIB_COMPONENTS
  VERSION_VAR OpenEXR_VERSION
  HANDLE_COMPONENTS
)

IF ( OpenEXR_FOUND )
  SET ( OpenEXR_LIBRARIES ${OpenEXR_LIB_COMPONENTS} )

  # We have to add both include and include/OpenEXR to the include
  # path in case OpenEXR and IlmBase are installed separately

  SET ( OpenEXR_INCLUDE_DIRS )
  LIST ( APPEND OpenEXR_INCLUDE_DIRS
    ${OpenEXR_INCLUDE_DIR}/../
    ${OpenEXR_INCLUDE_DIR}
    )
  SET ( OpenEXR_DEFINITIONS ${PC_OpenEXR_CFLAGS_OTHER} )

  SET ( OpenEXR_LIBRARY_DIRS )
  FOREACH ( LIB ${OpenEXR_LIB_COMPONENTS} )
    GET_FILENAME_COMPONENT ( _OPENEXR_LIBDIR ${LIB} DIRECTORY )
    LIST ( APPEND OpenEXR_LIBRARY_DIRS ${_OPENEXR_LIBDIR} )
  ENDFOREACH ()
  LIST ( REMOVE_DUPLICATES OpenEXR_LIBRARY_DIRS )

  # Configure imported target

  FOREACH ( COMPONENT ${OpenEXR_FIND_COMPONENTS} )
    IF ( NOT TARGET OpenEXR::${COMPONENT} )
      ADD_LIBRARY ( OpenEXR::${COMPONENT} UNKNOWN IMPORTED )
      SET_TARGET_PROPERTIES ( OpenEXR::${COMPONENT} PROPERTIES
        IMPORTED_LOCATION "${OpenEXR_${COMPONENT}_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${OpenEXR_DEFINITIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpenEXR_INCLUDE_DIRS}"
      )
    ENDIF ()
  ENDFOREACH ()
ELSEIF ( OpenEXR_FIND_REQUIRED )
  MESSAGE ( FATAL_ERROR "Unable to find OpenEXR" )
ENDIF ()
