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
-----------

Find OpenVDB include dirs, libraries and settings

Use this module by invoking find_package with the form::

  find_package(OpenVDB
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if OpenVDB is not found
    [COMPONENTS <libs>...] # OpenVDB libraries by their canonical name
                           # e.g. "openvdb" for "libopenvdb"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``OpenVDB::openvdb``
  The core openvdb library target.

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
``OpenVDB_USES_BLOSC``
  True if the OpenVDB Library has been built with blosc support
``OpenVDB_USES_LOG4CPLUS``
  True if the OpenVDB Library has been built with log4cplus support
``OpenVDB_USES_EXR``
  True if the OpenVDB Library has been built with openexr support
``OpenVDB_ABI``
  Set if this module was able to determine the ABI number the located
  OpenVDB Library was built against. Unset otherwise.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OpenVDB_INCLUDE_DIR``
  The directory containing ``openvdb/version.h``.
``OpenVDB_{COMPONENT}_LIBRARY``
  Individual component libraries for OpenVDB

Hints
^^^^^

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

CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )

# Support new if() IN_LIST operator
IF ( POLICY CMP0057 )
  CMAKE_POLICY ( SET CMP0057 NEW )
ENDIF ()

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
PKG_CHECK_MODULES ( PC_OpenVDB QUIET OpenVDB )

# Set various variables to track directories if this is being called from
# an installed location and from another package. The expected installation
# directory structure is:
#  <root>/lib/cmake/OpenVDB/FindOpenVDB.cmake
#  <root>/include
#  <root>/bin
# Note that _ROOT, _INCLUDEDIR and _LIBRARYDIR still take precedence if
# specified

GET_FILENAME_COMPONENT ( _IMPORT_PREFIX ${CMAKE_CURRENT_LIST_DIR} DIRECTORY )
GET_FILENAME_COMPONENT ( _IMPORT_PREFIX ${_IMPORT_PREFIX} DIRECTORY )
GET_FILENAME_COMPONENT ( _IMPORT_PREFIX ${_IMPORT_PREFIX} DIRECTORY )
SET ( _OPENVDB_INSTALLED_INCLUDE_DIR ${_IMPORT_PREFIX}/include )
SET ( _OPENVDB_INSTALLED_LIB_DIR ${_IMPORT_PREFIX}/lib )
SET ( _OPENVDB_INSTALLED_BIN_DIR ${_IMPORT_PREFIX}/bin )
UNSET ( _IMPORT_PREFIX )

# ------------------------------------------------------------------------
#  Search for OpenVDB include DIR
# ------------------------------------------------------------------------

SET ( _OPENVDB_INCLUDE_SEARCH_DIRS "" )
LIST ( APPEND _OPENVDB_INCLUDE_SEARCH_DIRS
  ${OPENVDB_INCLUDEDIR}
  ${_OPENVDB_ROOT_SEARCH_DIR}
  ${_OPENVDB_INSTALLED_INCLUDE_DIR}
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
  ${_OPENVDB_INSTALLED_LIB_DIR}
  ${PC_OpenVDB_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Build suffix directories

SET ( OPENVDB_PATH_SUFFIXES
  lib64
  lib
)

# Static library setup
IF ( UNIX AND OPENVDB_USE_STATIC_LIBS )
  SET ( _OPENVDB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
  SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".a" )
ENDIF ()

SET ( OpenVDB_LIB_COMPONENTS "" )

FOREACH ( COMPONENT ${OpenVDB_FIND_COMPONENTS} )
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

IF ( UNIX AND OPENVDB_USE_STATIC_LIBS )
  SET ( CMAKE_FIND_LIBRARY_SUFFIXES ${_OPENVDB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES} )
  UNSET ( _OPENVDB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES )
ENDIF ()

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

IF ( NOT OpenVDB_FOUND )
  IF ( OpenVDB_FIND_REQUIRED )
    MESSAGE ( FATAL_ERROR "Unable to find OpenVDB" )
  ENDIF ()
  RETURN ()
ENDIF ()

# ------------------------------------------------------------------------
#  Determine ABI number
# ------------------------------------------------------------------------

# Set the ABI number the library was built against. Uses vdb_print

IF ( EXISTS ${_OPENVDB_INSTALLED_BIN_DIR} )
  SET ( _OPENVDB_INSTALLED_PRINT_BIN "${_OPENVDB_INSTALLED_BIN_DIR}/vdb_print" )
  IF ( EXISTS ${_OPENVDB_INSTALLED_PRINT_BIN} )
    SET ( _VDB_PRINT_VERSION_STRING "" )
    SET ( _VDB_PRINT_RETURN_STATUS "" )

    EXECUTE_PROCESS ( COMMAND ${_OPENVDB_INSTALLED_PRINT_BIN} "--version"
      RESULT_VARIABLE _VDB_PRINT_RETURN_STATUS
      OUTPUT_VARIABLE _VDB_PRINT_VERSION_STRING
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    IF ( NOT ${_VDB_PRINT_RETURN_STATUS} )
      SET ( OpenVDB_ABI )
      STRING ( REGEX REPLACE ".*abi([0-9]*).*" "\\1" OpenVDB_ABI ${_VDB_PRINT_VERSION_STRING} )
    ENDIF ()
  ENDIF ()
ENDIF ()

IF ( NOT OpenVDB_FIND_QUIET )
  IF ( NOT OpenVDB_ABI )
    MESSAGE ( WARNING "Unable to determine OpenVDB ABI version from OpenVDB installation. The "
      "library major version \"${OpenVDB_MAJOR_VERSION}\" will be inferred. If this is not correct, "
      "use add_definitions(-DOPENVDB_ABI_VERSION_NUMBER=N)"
      )
  ELSE ()
    MESSAGE ( STATUS "OpenVDB ABI Version: ${OpenVDB_ABI}" )
  ENDIF ()
ENDIF ()

# ------------------------------------------------------------------------
#  Handle OpenVDB dependencies
# ------------------------------------------------------------------------

# Add standard dependencies

FIND_PACKAGE ( IlmBase REQUIRED COMPONENTS Half )
FIND_PACKAGE ( TBB REQUIRED COMPONENTS tbb )
FIND_PACKAGE ( ZLIB REQUIRED )
FIND_PACKAGE ( Boost REQUIRED COMPONENTS iostreams system )

# Use GetPrerequisites to see which libraries this OpenVDB lib has linked to
# which we can query for optional deps. This basically runs ldd/otoll/objdump
# etc to track deps. We could use a vdb_config binary tools here to improve
# this process

INCLUDE ( GetPrerequisites )

SET ( _EXCLUDE_SYSTEM_PREREQUISITES 1 )
SET ( _RECURSE_PREREQUISITES 0 )
SET ( _OPENVDB_PREREQUISITE_LIST )

GET_PREREQUISITES ( ${OpenVDB_openvdb_LIBRARY}
  _OPENVDB_PREREQUISITE_LIST
  ${_EXCLUDE_SYSTEM_PREREQUISITES}
  ${_RECURSE_PREREQUISITES}
  ""
  "${SYSTEM_LIBRARY_PATHS}"
)

UNSET ( _EXCLUDE_SYSTEM_PREREQUISITES )
UNSET ( _RECURSE_PREREQUISITES )

# As the way we resolve optional libraries relies on library file names, use
# the configuration options from the main CMakeLists.txt to allow users
# to manually identify the requirements of OpenVDB builds if they know them.

SET ( OpenVDB_USES_BLOSC ${USE_BLOSC} )
SET ( OpenVDB_USES_LOG4CPLUS ${USE_LOG4CPLUS} )
SET ( OpenVDB_USES_EXR ${USE_EXR} )

# Search for optional dependencies

FOREACH ( PREREQUISITE ${_OPENVDB_PREREQUISITE_LIST} )
  SET ( _HAS_DEP )
  GET_FILENAME_COMPONENT ( PREREQUISITE ${PREREQUISITE} NAME )

  STRING ( FIND ${PREREQUISITE} "blosc" _HAS_DEP )
  IF ( NOT ${_HAS_DEP} EQUAL -1 )
    SET ( OpenVDB_USES_BLOSC ON )
  ENDIF ()

  STRING ( FIND ${PREREQUISITE} "log4cplus" _HAS_DEP )
  IF ( NOT ${_HAS_DEP} EQUAL -1 )
    SET ( OpenVDB_USES_LOG4CPLUS ON )
  ENDIF ()

  STRING ( FIND ${PREREQUISITE} "IlmImf" _HAS_DEP )
  IF ( NOT ${_HAS_DEP} EQUAL -1 )
    SET ( OpenVDB_USES_EXR ON )
  ENDIF ()
ENDFOREACH ()

UNSET ( _OPENVDB_PREREQUISITE_LIST )
UNSET ( _HAS_DEP )

IF ( OpenVDB_USES_BLOSC )
  FIND_PACKAGE ( Blosc REQUIRED )
ENDIF ()

IF ( OpenVDB_USES_LOG4CPLUS )
  FIND_PACKAGE ( Log4cplus REQUIRED )
ENDIF ()

IF ( OpenVDB_USES_EXR )
  FIND_PACKAGE ( IlmBase REQUIRED )
  FIND_PACKAGE ( OpenEXR REQUIRED )
ENDIF ()

IF ( UNIX )
  FIND_PACKAGE ( Threads REQUIRED )
ENDIF ()

# Set deps. Note that the order here is important. If we're building against
# Houdini 17.5 we must include OpenEXR and IlmBase deps first to ensure the
# users chosen namespaced headers are correctly prioritized. Otherwise other
# include paths from shared installs (including houdini) may pull in the wrong
# headers

SET ( _OPENVDB_VISIBLE_DEPENDENCIES
  Boost::iostreams
  Boost::system
  IlmBase::Half
  )

SET ( _OPENVDB_DEFINITIONS )
IF ( OpenVDB_ABI )
  LIST ( APPEND _OPENVDB_DEFINITIONS "-DOPENVDB_ABI_VERSION_NUMBER=${OpenVDB_ABI}" )
ENDIF ()

IF ( OpenVDB_USES_EXR )
  LIST ( APPEND _OPENVDB_VISIBLE_DEPENDENCIES
    IlmBase::IlmThread
    IlmBase::Iex
    IlmBase::Imath
    OpenEXR::IlmImf
    )
  LIST ( APPEND _OPENVDB_DEFINITIONS "-DOPENVDB_TOOLS_RAYTRACER_USE_EXR" )
ENDIF ()

IF ( OpenVDB_USES_LOG4CPLUS )
  LIST ( APPEND _OPENVDB_VISIBLE_DEPENDENCIES Log4cplus::log4cplus )
  LIST ( APPEND _OPENVDB_DEFINITIONS "-DOPENVDB_USE_LOG4CPLUS" )
ENDIF ()

LIST ( APPEND _OPENVDB_VISIBLE_DEPENDENCIES
  TBB::tbb
)
IF ( UNIX )
  LIST ( APPEND _OPENVDB_VISIBLE_DEPENDENCIES
    Threads::Threads
  )
ENDIF ()

SET ( _OPENVDB_HIDDEN_DEPENDENCIES )

IF ( OpenVDB_USES_BLOSC )
  LIST ( APPEND _OPENVDB_HIDDEN_DEPENDENCIES Blosc::blosc )
ENDIF ()

LIST ( APPEND _OPENVDB_HIDDEN_DEPENDENCIES ZLIB::ZLIB )

# ------------------------------------------------------------------------
#  Configure imported target
# ------------------------------------------------------------------------

SET ( OpenVDB_LIBRARIES
  ${OpenVDB_LIB_COMPONENTS}
)
SET ( OpenVDB_INCLUDE_DIRS ${OpenVDB_INCLUDE_DIR} )

SET ( OpenVDB_DEFINITIONS )
LIST ( APPEND OpenVDB_DEFINITIONS "${PC_OpenVDB_CFLAGS_OTHER}" )
LIST ( APPEND OpenVDB_DEFINITIONS "${_OPENVDB_DEFINITIONS}" )
LIST ( REMOVE_DUPLICATES OpenVDB_DEFINITIONS )

SET ( OpenVDB_LIBRARY_DIRS "" )
FOREACH ( LIB ${OpenVDB_LIB_COMPONENTS} )
  GET_FILENAME_COMPONENT ( _OPENVDB_LIBDIR ${LIB} DIRECTORY )
  LIST ( APPEND OpenVDB_LIBRARY_DIRS ${_OPENVDB_LIBDIR} )
ENDFOREACH ()
LIST ( REMOVE_DUPLICATES OpenVDB_LIBRARY_DIRS )

FOREACH ( COMPONENT ${OpenVDB_FIND_COMPONENTS} )
  IF ( NOT TARGET OpenVDB::${COMPONENT} )
    ADD_LIBRARY ( OpenVDB::${COMPONENT} UNKNOWN IMPORTED )
    SET_TARGET_PROPERTIES ( OpenVDB::${COMPONENT} PROPERTIES
      IMPORTED_LOCATION "${OpenVDB_${COMPONENT}_LIBRARY}"
      INTERFACE_COMPILE_OPTIONS "${OpenVDB_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${OpenVDB_INCLUDE_DIR}"
      IMPORTED_LINK_DEPENDENT_LIBRARIES "${_OPENVDB_HIDDEN_DEPENDENCIES}" # non visible deps
      INTERFACE_LINK_LIBRARIES "${_OPENVDB_VISIBLE_DEPENDENCIES}" # visible deps (headers)
    )
  ENDIF ()
ENDFOREACH ()

UNSET ( _OPENVDB_DEFINITIONS )
UNSET ( _OPENVDB_VISIBLE_DEPENDENCIES )
UNSET ( _OPENVDB_HIDDEN_DEPENDENCIES )
