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

FindILMBase
---------

Find ILMBase include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(ILMBase
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if ILMBase is not found
    [COMPONENTS <libs>...] # ILMBase libraries by their canonical name
                           # e.g. "Half" for "libHalf"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``ILMBase::Half``
  The Half library target.
``ILMBase::Iex``
  The Iex library target.
``ILMBase::IexMath``
  The IexMath library target.
``ILMBase::IlmThread``
  The IlmThread library target.
``ILMBase::Imath``
  The Imath library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``ILMBase_FOUND``
  True if the system has the ILMBase library.
``ILMBase_VERSION``
  The version of the ILMBase library which was found.
``ILMBase_INCLUDE_DIRS``
  Include directories needed to use ILMBase.
``ILMBase_LIBRARIES``
  Libraries needed to link to ILMBase.
``ILMBase_LIBRARY_DIRS``
  ILMBase library directories.
``ILMBase_{COMPONENT}_FOUND``
  True if the system has the named ILMBase component.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``ILMBase_INCLUDE_DIR``
  The directory containing ``ILMBase/config-auto.h``.
``ILMBase_{COMPONENT}_LIBRARY``
  Individual component libraries for ILMBase
``ILMBase_{COMPONENT}_DLL``
  Individual component dlls for ILMBase on Windows.

Hints
^^^^^^^^^^^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``ILMBASE_ROOT``
  Preferred installation prefix.
``ILMBASE_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``ILMBASE_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

MARK_AS_ADVANCED (
  ILMBase_INCLUDE_DIR
  ILMBase_LIBRARY
  ILMBASE_NAMESPACE_VERSIONING
)

SET ( _ILMBASE_COMPONENT_LIST
  Half
  Iex
  IexMath
  IlmThread
  Imath
  )

IF ( ILMBase_FIND_COMPONENTS )
  SET ( ILMBASE_COMPONENTS_PROVIDED TRUE )
  SET ( _IGNORED_COMPONENTS "" )
  FOREACH ( COMPONENT ${ILMBase_FIND_COMPONENTS} )
    IF ( NOT ${COMPONENT} IN_LIST _ILMBASE_COMPONENT_LIST )
      LIST ( APPEND _IGNORED_COMPONENTS ${COMPONENT} )
    ENDIF ()
  ENDFOREACH()

  IF ( _IGNORED_COMPONENTS )
    MESSAGE ( STATUS "Ignoring unknown components of ILMBase:" )
    FOREACH ( COMPONENT ${_IGNORED_COMPONENTS} )
      MESSAGE ( STATUS "  ${COMPONENT}" )
    ENDFOREACH ()
    LIST ( REMOVE_ITEM ILMBase_FIND_COMPONENTS ${_IGNORED_COMPONENTS} )
  ENDIF ()
ELSE ()
  SET ( ILMBASE_COMPONENTS_PROVIDED FALSE )
  SET ( ILMBase_FIND_COMPONENTS ${_ILMBASE_COMPONENT_LIST} )
ENDIF ()

# Append ILMBASE_ROOT or $ENV{ILMBASE_ROOT} if set (prioritize the direct cmake var)
SET ( _ILMBASE_ROOT_SEARCH_DIR "" )

IF ( ILMBASE_ROOT )
  LIST ( APPEND _ILMBASE_ROOT_SEARCH_DIR ${ILMBASE_ROOT} )
ELSE ()
  SET ( _ENV_ILMBASE_ROOT $ENV{ILMBASE_ROOT} )
  IF ( _ENV_ILMBASE_ROOT )
    LIST ( APPEND _ILMBASE_ROOT_SEARCH_DIR ${_ENV_ILMBASE_ROOT} )
  ENDIF ()
ENDIF ()

# Additionally try and use pkconfig to find ILMBase

FIND_PACKAGE ( PkgConfig )
PKG_CHECK_MODULES ( PC_ILMBase QUIET ilmbase )

# ------------------------------------------------------------------------
#  Search for ILMBase include DIR
# ------------------------------------------------------------------------

SET ( _ILMBASE_INCLUDE_SEARCH_DIRS "" )
LIST ( APPEND _ILMBASE_INCLUDE_SEARCH_DIRS
  ${ILMBASE_INCLUDEDIR}
  ${_ILMBASE_ROOT_SEARCH_DIR}
  ${PC_ILMBase_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Look for a standard OpenEXR header file.
FIND_PATH ( ILMBase_INCLUDE_DIR IlmBaseConfig.h
  NO_DEFAULT_PATH
  PATHS ${_ILMBASE_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include/OpenEXR OpenEXR
  )

IF ( EXISTS "${ILMBase_INCLUDE_DIR}/IlmBaseConfig.h" )
  # Get the ILMBASE version information from the config header
  FILE ( STRINGS "${ILMBase_INCLUDE_DIR}/IlmBaseConfig.h"
    _ilmbase_version_major_string REGEX "#define ILMBASE_VERSION_MAJOR "
    )
  STRING ( REGEX REPLACE "#define ILMBASE_VERSION_MAJOR" ""
    _ilmbase_version_major_string "${_ilmbase_version_major_string}"
    )
  STRING ( STRIP "${_ilmbase_version_major_string}" ILMBase_VERSION_MAJOR )

  FILE ( STRINGS "${ILMBase_INCLUDE_DIR}/IlmBaseConfig.h"
     _ilmbase_version_minor_string REGEX "#define ILMBASE_VERSION_MINOR "
    )
  STRING ( REGEX REPLACE "#define ILMBASE_VERSION_MINOR" ""
    _ilmbase_version_minor_string "${_ilmbase_version_minor_string}"
    )
  STRING ( STRIP "${_ilmbase_version_minor_string}" ILMBase_VERSION_MINOR )

  UNSET ( _ilmbase_version_major_string )
  UNSET ( _ilmbase_version_minor_string )

  SET ( ILMBase_VERSION ${ILMBase_VERSION_MAJOR}.${ILMBase_VERSION_MINOR} )
ENDIF ()

# ------------------------------------------------------------------------
#  Search for ILMBASE lib DIR
# ------------------------------------------------------------------------

SET ( _ILMBASE_LIBRARYDIR_SEARCH_DIRS "" )

# Append to _ILMBASE_LIBRARYDIR_SEARCH_DIRS in priority order

LIST ( APPEND _ILMBASE_LIBRARYDIR_SEARCH_DIRS
  ${ILMBASE_LIBRARYDIR}
  ${_ILMBASE_ROOT_SEARCH_DIR}
  ${PC_ILMBase_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Build suffix directories

SET ( ILMBASE_PATH_SUFFIXES
  lib64
  lib
)

IF ( ${CMAKE_CXX_COMPILER_ID} STREQUAL GNU )
  LIST ( INSERT ILMBASE_PATH_SUFFIXES 0 lib/x86_64-linux-gnu )
ENDIF ()

SET ( _ILMBASE_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( ILMBase_LIB_COMPONENTS "" )

FOREACH ( COMPONENT ${ILMBase_FIND_COMPONENTS} )
  # library suffix handling
  IF ( WIN32 )
    SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".lib" )
  ENDIF ()

  IF ( ILMBASE_USE_STATIC_LIBS )
    IF ( UNIX )
      SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".a" )
    ENDIF ()
  ENDIF ()

  SET ( LIB_NAME ${COMPONENT} )
  IF ( ILMBASE_NAMESPACE_VERSIONING AND ILMBase_VERSION )
    IF ( NOT ${COMPONENT} STREQUAL "Half" )
      SET ( LIB_NAME "${LIB_NAME}-${ILMBase_VERSION_MAJOR}_${ILMBase_VERSION_MINOR}" )
    ENDIF ()
  ENDIF ()

  FIND_LIBRARY ( ILMBase_${COMPONENT}_LIBRARY ${LIB_NAME}
    NO_DEFAULT_PATH
    PATHS ${_ILMBASE_LIBRARYDIR_SEARCH_DIRS}
    PATH_SUFFIXES ${ILMBASE_PATH_SUFFIXES}
    )
  LIST ( APPEND ILMBase_LIB_COMPONENTS ${ILMBase_${COMPONENT}_LIBRARY} )

  IF ( NOT ILMBASE_USE_STATIC_LIBS AND WIN32 )
    SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".dll" )
    FIND_LIBRARY ( ILMBase_${COMPONENT}_DLL ${LIB_NAME}
      NO_DEFAULT_PATH
      PATHS ${_ILMBASE_LIBRARYDIR_SEARCH_DIRS}
      PATH_SUFFIXES bin
      )
  ENDIF ()

  IF ( ILMBase_${COMPONENT}_LIBRARY )
    SET ( ILMBase_${COMPONENT}_FOUND TRUE )
  ELSE ()
    SET ( ILMBase_${COMPONENT}_FOUND FALSE )
  ENDIF ()
ENDFOREACH ()

# reset lib suffix

SET ( CMAKE_FIND_LIBRARY_SUFFIXES ${_ILMBASE_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES} )

# ------------------------------------------------------------------------
#  Cache and set ILMBASE_FOUND
# ------------------------------------------------------------------------

INCLUDE ( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( ILMBase
  FOUND_VAR ILMBase_FOUND
  REQUIRED_VARS
    ILMBase_INCLUDE_DIR
    ILMBase_LIB_COMPONENTS
  VERSION_VAR ILMBase_VERSION
  HANDLE_COMPONENTS
)

IF ( ILMBase_FOUND )
  SET ( ILMBase_LIBRARIES ${ILMBase_LIB_COMPONENTS} )

  # We have to add both include and include/OpenEXR to the include
  # path in case OpenEXR and ILMBase are installed separately
  GET_FILENAME_COMPONENT ( ILMBase_INCLUDE_DIR ${ILMBase_INCLUDE_DIR} DIRECTORY )

  SET ( ILMBase_INCLUDE_DIRS )
  LIST ( APPEND ILMBase_INCLUDE_DIRS
    ${ILMBase_INCLUDE_DIR}
    ${ILMBase_INCLUDE_DIR}/OpenEXR
    )
  SET ( ILMBase_DEFINITIONS ${PC_ILMBase_CFLAGS_OTHER} )

  SET ( ILMBase_LIBRARY_DIRS "" )
  FOREACH ( LIB ${ILMBase_LIB_COMPONENTS} )
    GET_FILENAME_COMPONENT ( _ILMBASE_LIBDIR ${LIB} DIRECTORY )
    LIST ( APPEND ILMBase_LIBRARY_DIRS ${_ILMBASE_LIBDIR} )
  ENDFOREACH ()
  LIST ( REMOVE_DUPLICATES ILMBase_LIBRARY_DIRS )

  # Configure imported targets

  FOREACH ( COMPONENT ${ILMBase_FIND_COMPONENTS} )
    IF ( NOT TARGET ILMBase::${COMPONENT} )
      ADD_LIBRARY ( ILMBase::${COMPONENT} UNKNOWN IMPORTED )
      SET_TARGET_PROPERTIES ( ILMBase::${COMPONENT} PROPERTIES
        IMPORTED_LOCATION "${ILMBase_${COMPONENT}_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${ILMBase_DEFINITIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${ILMBase_INCLUDE_DIRS}"
      )
    ENDIF ()
  ENDFOREACH ()

ELSEIF ( ILMBase_FIND_REQUIRED )
  MESSAGE ( FATAL_ERROR "Unable to find ILMBase")
ENDIF ()
