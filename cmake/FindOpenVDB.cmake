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

FindOpenVDB
---------

Find OpenVDB include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(OpenVDB
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if OpenVDB is not found
    [COMPONENTS <libs>...] # OpenVDB libraries by their canonical name
                           # e.g. "IlmImf" for "libIlmImf"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``OpenVDB::openvdb``
  The core openvdb library target.
``OpenVDB::pyopenvdb``
  The python openvdb library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``OpenVDB_FOUND``
  True if the system has the OpenVDB library.
``OpenVDB_VERSION``
  The version of the OpenVDB library which was found.
``OpenVDB_INCLUDE_DIRS``
  Include directories needed to use OpenVDB.
``OpenVDB_LIBRARIES``
  Libraries needed to link to OpenVDB.
``OpenVDB_LIBRARY_DIRS``
  OpenVDB library directories.
``OpenVDB_DEFINITIONS``
  Definitions to use when compiling code that uses OpenVDB.
``OpenVDB_{COMPONENT}_FOUND``
  True if the system has the named OpenVDB component.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OpenVDB_INCLUDE_DIR``
  The directory containing ``OpenVDB/config-auto.h``.
``OpenVDB_{COMPONENT}_LIBRARY``
  Individual component libraries for OpenVDB

Hints
^^^^^^^^^^^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``OPENVDB_ROOT``
  Preferred installation prefix.
``OPENVDB_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``OPENVDB_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

MARK_AS_ADVANCED (
  OpenVDB_INCLUDE_DIR
  OpenVDB_LIBRARY
  OPENVDB_NAMESPACE_VERSIONING
)

SET ( _OPENVDB_COMPONENT_LIST
  openvdb
  )

IF ( OpenVDB_FIND_COMPONENTS )
  SET ( OPENVDB_COMPONENTS_PROVIDED TRUE )
  SET ( _IGNORED_COMPONENTS "" )
  FOREACH ( COMPONENT ${OpenVDB_FIND_COMPONENTS} )
    IF ( NOT ${COMPONENT} IN_LIST _OPENVDB_COMPONENT_LIST )
      LIST ( APPEND _IGNORED_COMPONENTS ${COMPONENT} )
    ENDIF ()
  ENDFOREACH()

  IF ( _IGNORED_COMPONENTS )
    MESSAGE ( STATUS "Ignoring unknown components of OpenVDB:" )
    FOREACH ( COMPONENT ${_IGNORED_COMPONENTS} )
      MESSAGE ( STATUS "  ${COMPONENT}" )
    ENDFOREACH ()
    LIST ( REMOVE_ITEM OpenVDB_FIND_COMPONENTS ${_IGNORED_COMPONENTS} )
  ENDIF ()
ELSE ()
  SET ( OPENVDB_COMPONENTS_PROVIDED FALSE )
  SET ( OpenVDB_FIND_COMPONENTS ${_OPENVDB_COMPONENT_LIST} )
ENDIF ()

# Append OPENVDB_ROOT or $ENV{OPENVDB_ROOT} if set (prioritize the direct cmake var)
SET ( _OPENVDB_ROOT_SEARCH_DIR "" )

IF ( OPENVDB_ROOT )
  LIST ( APPEND _OPENVDB_ROOT_SEARCH_DIR ${OPENVDB_ROOT} )
ELSE ()
  SET ( _ENV_OPENVDB_ROOT $ENV{OPENVDB_ROOT} )
  IF ( _ENV_OPENVDB_ROOT )
    LIST ( APPEND _OPENVDB_ROOT_SEARCH_DIR ${_ENV_OPENVDB_ROOT} )
  ENDIF ()
ENDIF ()

# Additionally try and use pkconfig to find OpenVDB

FIND_PACKAGE ( PkgConfig )
PKG_CHECK_MODULES ( PC_OpenVDB QUIET openvdb )

# ------------------------------------------------------------------------
#  Search for OpenVDB include DIR
# ------------------------------------------------------------------------

SET ( _OPENVDB_INCLUDE_SEARCH_DIRS "" )
LIST ( APPEND _OPENVDB_INCLUDE_SEARCH_DIRS
  ${OPENVDB_INCLUDEDIR}
  ${_OPENVDB_ROOT_SEARCH_DIR}
  ${PC_OpenVDB_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Look for a standard OpenVDB header file.
FIND_PATH ( OpenVDB_INCLUDE_DIR openvdb/version.h
  NO_DEFAULT_PATH
  PATHS ${_OPENVDB_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include
  )

IF ( EXISTS "${OpenVDB_INCLUDE_DIR}/openvdb/version.h" )
  SET ( OPENVDB_VERSION_FILE ${OpenVDB_INCLUDE_DIR}/openvdb/version.h )
  FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+.*"
    )
  STRING ( REGEX REPLACE "^.*OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    OpenVDB_MAJOR_VERSION "${openvdb_version_str}"
    )

  FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+.*"
    )
  STRING ( REGEX REPLACE "^.*OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    OpenVDB_MINOR_VERSION "${openvdb_version_str}"
    )

  FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+.*"
    )
  STRING ( REGEX REPLACE "^.*OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    OpenVDB_PATCH_VERSION "${openvdb_version_str}"
    )
  UNSET ( openvdb_version_str )
  UNSET ( OPENVDB_VERSION_FILE )
  SET ( OpenVDB_VERSION ${OpenVDB_MAJOR_VERSION}.${OpenVDB_MINOR_VERSION}.${OpenVDB_PATCH_VERSION} )
ENDIF ()

# ------------------------------------------------------------------------
#  Search for OPENVDB lib DIR
# ------------------------------------------------------------------------

SET ( _OPENVDB_LIBRARYDIR_SEARCH_DIRS "" )

# Append to _OPENVDB_LIBRARYDIR_SEARCH_DIRS in priority order

LIST ( APPEND _OPENVDB_LIBRARYDIR_SEARCH_DIRS
  ${OPENVDB_LIBRARYDIR}
  ${_OPENVDB_ROOT_SEARCH_DIR}
  ${PC_OpenVDB_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Build suffix directories

SET ( OPENVDB_PATH_SUFFIXES
  lib64
  lib
)

SET ( _OPENVDB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( OpenVDB_LIB_COMPONENTS "" )

FOREACH ( COMPONENT ${OpenVDB_FIND_COMPONENTS} )
  IF ( OPENVDB_USE_STATIC_LIBS )
    IF ( UNIX )
      SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".a" )
    ENDIF ()
  ENDIF ()

  SET ( LIB_NAME ${COMPONENT} )
  FIND_LIBRARY ( OpenVDB_${COMPONENT}_LIBRARY ${LIB_NAME}
    NO_DEFAULT_PATH
    PATHS ${_OPENVDB_LIBRARYDIR_SEARCH_DIRS}
    PATH_SUFFIXES ${OPENVDB_PATH_SUFFIXES}
    )
  LIST ( APPEND OpenVDB_LIB_COMPONENTS ${OpenVDB_${COMPONENT}_LIBRARY} )

  IF ( OpenVDB_${COMPONENT}_LIBRARY )
    SET ( OpenVDB_${COMPONENT}_FOUND TRUE )
  ELSE ()
    SET ( OpenVDB_${COMPONENT}_FOUND FALSE )
  ENDIF ()
ENDFOREACH ()

# reset lib suffix

SET ( CMAKE_FIND_LIBRARY_SUFFIXES ${_OPENVDB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES} )

# ------------------------------------------------------------------------
#  Cache and set OPENVDB_FOUND
# ------------------------------------------------------------------------

INCLUDE ( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( OpenVDB
  FOUND_VAR OpenVDB_FOUND
  REQUIRED_VARS
    OpenVDB_INCLUDE_DIR
    OpenVDB_LIB_COMPONENTS
  VERSION_VAR OpenVDB_VERSION
  HANDLE_COMPONENTS
)

IF ( OpenVDB_FOUND )
  SET ( OpenVDB_LIBRARIES
    ${OpenVDB_LIB_COMPONENTS}
  )
  SET ( OpenVDB_INCLUDE_DIRS ${OpenVDB_INCLUDE_DIR} )
  SET ( OpenVDB_DEFINITIONS ${PC_OpenVDB_CFLAGS_OTHER} )

  SET ( OpenVDB_LIBRARY_DIRS "" )
  FOREACH ( LIB ${OpenVDB_LIB_COMPONENTS} )
    GET_FILENAME_COMPONENT ( _OPENVDB_LIBDIR ${LIB} DIRECTORY )
    LIST ( APPEND OpenVDB_LIBRARY_DIRS ${_OPENVDB_LIBDIR} )
  ENDFOREACH ()
  LIST ( REMOVE_DUPLICATES OpenVDB_LIBRARY_DIRS )

  # Configure imported target

  FOREACH ( COMPONENT ${OpenVDB_FIND_COMPONENTS} )
    IF ( NOT TARGET OpenVDB::${COMPONENT} )
      ADD_LIBRARY ( OpenVDB::${COMPONENT} UNKNOWN IMPORTED )
      SET_TARGET_PROPERTIES ( OpenVDB::${COMPONENT} PROPERTIES
        IMPORTED_LOCATION "${OpenVDB_${COMPONENT}_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${OpenVDB_DEFINITIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpenVDB_INCLUDE_DIR}"
      )
    ENDIF ()
  ENDFOREACH ()
ELSEIF ( OpenVDB_FIND_REQUIRED )
  MESSAGE ( FATAL_ERROR "Unable to find OpenVDB" )
ENDIF ()
