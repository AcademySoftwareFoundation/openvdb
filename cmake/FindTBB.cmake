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

FindTBB
-------

Find Tbb include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(TBB
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if Tbb is not found
    [COMPONENTS <libs>...] # Tbb libraries by their canonical name
                           # e.g. "tbb" for "libtbb"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``TBB::tbb``
  The tbb library target.
``TBB::tbbmalloc``
  The tbbmalloc library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Tbb_FOUND``
  True if the system has the Tbb library.
``Tbb_VERSION``
  The version of the Tbb library which was found.
``Tbb_INCLUDE_DIRS``
  Include directories needed to use Tbb.
``Tbb_LIBRARIES``
  Libraries needed to link to Tbb.
``Tbb_LIBRARY_DIRS``
  Tbb library directories.
``TBB_{COMPONENT}_FOUND``
  True if the system has the named TBB component.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Tbb_INCLUDE_DIR``
  The directory containing ``tbb/tbb_stddef.h``.
``Tbb_{COMPONENT}_LIBRARY``
  Individual component libraries for Tbb

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``TBB_ROOT``
  Preferred installation prefix.
``TBB_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``TBB_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

MARK_AS_ADVANCED (
  Tbb_INCLUDE_DIR
  Tbb_LIBRARY
)

SET ( _TBB_COMPONENT_LIST
  tbb
  tbbmalloc
  )

IF ( TBB_FIND_COMPONENTS )
  SET ( _TBB_COMPONENTS_PROVIDED TRUE )
  SET ( _IGNORED_COMPONENTS "" )
  FOREACH ( COMPONENT ${TBB_FIND_COMPONENTS} )
    IF ( NOT ${COMPONENT} IN_LIST _TBB_COMPONENT_LIST )
      LIST ( APPEND _IGNORED_COMPONENTS ${COMPONENT} )
    ENDIF ()
  ENDFOREACH()

  IF ( _IGNORED_COMPONENTS )
    MESSAGE ( STATUS "Ignoring unknown components of TBB:" )
    FOREACH ( COMPONENT ${_IGNORED_COMPONENTS} )
      MESSAGE ( STATUS "  ${COMPONENT}" )
    ENDFOREACH ()
    LIST ( REMOVE_ITEM TBB_FIND_COMPONENTS ${_IGNORED_COMPONENTS} )
  ENDIF ()
ELSE ()
  SET ( _TBB_COMPONENTS_PROVIDED FALSE )
  SET ( TBB_FIND_COMPONENTS ${_TBB_COMPONENT_LIST} )
ENDIF ()

# Append TBB_ROOT or $ENV{TBB_ROOT} if set (prioritize the direct cmake var)
SET ( _TBB_ROOT_SEARCH_DIR "" )

IF ( TBB_ROOT )
  LIST ( APPEND _TBB_ROOT_SEARCH_DIR ${TBB_ROOT} )
ELSE ()
  SET ( _ENV_TBB_ROOT $ENV{TBB_ROOT} )
  IF ( _ENV_TBB_ROOT )
    LIST ( APPEND _TBB_ROOT_SEARCH_DIR ${_ENV_TBB_ROOT} )
  ENDIF ()
ENDIF ()

# Additionally try and use pkconfig to find Tbb

FIND_PACKAGE ( PkgConfig )
PKG_CHECK_MODULES ( PC_Tbb QUIET tbb )

# ------------------------------------------------------------------------
#  Search for tbb include DIR
# ------------------------------------------------------------------------

SET ( _TBB_INCLUDE_SEARCH_DIRS "" )
LIST ( APPEND _TBB_INCLUDE_SEARCH_DIRS
  ${TBB_INCLUDEDIR}
  ${_TBB_ROOT_SEARCH_DIR}
  ${PC_Tbb_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Look for a standard tbb header file.
FIND_PATH ( Tbb_INCLUDE_DIR tbb/tbb_stddef.h
  NO_DEFAULT_PATH
  PATHS ${_TBB_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include
  )

IF ( EXISTS "${Tbb_INCLUDE_DIR}/tbb/tbb_stddef.h" )
    FILE ( STRINGS "${Tbb_INCLUDE_DIR}/tbb/tbb_stddef.h"
      _tbb_version_major_string REGEX "#define TBB_VERSION_MAJOR "
      )
    STRING ( REGEX REPLACE "#define TBB_VERSION_MAJOR" ""
      _tbb_version_major_string "${_tbb_version_major_string}"
      )
    STRING ( STRIP "${_tbb_version_major_string}" Tbb_VERSION_MAJOR )

    FILE ( STRINGS "${Tbb_INCLUDE_DIR}/tbb/tbb_stddef.h"
       _tbb_version_minor_string REGEX "#define TBB_VERSION_MINOR "
      )
    STRING ( REGEX REPLACE "#define TBB_VERSION_MINOR" ""
      _tbb_version_minor_string "${_tbb_version_minor_string}"
      )
    STRING ( STRIP "${_tbb_version_minor_string}" Tbb_VERSION_MINOR )

    UNSET ( _tbb_version_major_string )
    UNSET ( _tbb_version_minor_string )

    SET ( Tbb_VERSION ${Tbb_VERSION_MAJOR}.${Tbb_VERSION_MINOR} )
ENDIF ()

# ------------------------------------------------------------------------
#  Search for TBB lib DIR
# ------------------------------------------------------------------------

SET ( _TBB_LIBRARYDIR_SEARCH_DIRS "" )

# Append to _TBB_LIBRARYDIR_SEARCH_DIRS in priority order

SET ( _TBB_LIBRARYDIR_SEARCH_DIRS "" )
LIST ( APPEND _TBB_LIBRARYDIR_SEARCH_DIRS
  ${TBB_LIBRARYDIR}
  ${_TBB_ROOT_SEARCH_DIR}
  ${PC_Tbb_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

SET ( TBB_PATH_SUFFIXES
  lib64
  lib
)

SET ( _TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
set ( Tbb_LIB_COMPONENTS "" )

# platform branching

IF ( UNIX )
  LIST ( INSERT TBB_PATH_SUFFIXES 0 lib/x86_64-linux-gnu )
ENDIF ()

IF (APPLE)
  IF (TBB_FOR_CLANG)
    LIST ( INSERT TBB_PATH_SUFFIXES 0 lib/libc++ )
  ENDIF ()
ELSEIF ( WIN32 )
  IF ( MSVC10 )
    SET ( TBB_VC_DIR vc10 )
  ELSEIF ( MSVC11 )
    SET ( TBB_VC_DIR vc11 )
  ELSEIF ( MSVC12 )
    SET ( TBB_VC_DIR vc12 )
  ENDIF ()
  LIST ( INSERT TBB_PATH_SUFFIXES 0 lib/intel64/${TBB_VC_DIR} )
ELSE ()
  IF ( ${CMAKE_CXX_COMPILER_ID} STREQUAL GNU )
    IF ( TBB_MATCH_COMPILER_VERSION )
      STRING ( REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${CMAKE_CXX_COMPILER_VERSION} )
      LIST ( GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR )
      LIST ( GET GCC_VERSION_COMPONENTS 1 GCC_MINOR )
      LIST ( INSERT TBB_PATH_SUFFIXES 0 lib/intel64/gcc${GCC_MAJOR}.${GCC_MINOR} )
    ELSE ()
      LIST ( INSERT TBB_PATH_SUFFIXES 0 lib/intel64/gcc4.4 )
    ENDIF ()
  ENDIF ()
ENDIF ()

FOREACH ( COMPONENT ${TBB_FIND_COMPONENTS} )
  # library suffix handling
  IF ( WIN32 )
    SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".lib" )
  ENDIF ()

  IF ( TBB_USE_STATIC_LIBS )
    IF ( UNIX )
      SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".a" )
    ENDIF ()
  ENDIF ()

  SET ( LIB_NAME ${COMPONENT} )
  FIND_LIBRARY ( Tbb_${COMPONENT}_LIBRARY ${LIB_NAME}
    NO_DEFAULT_PATH
    PATHS ${_TBB_LIBRARYDIR_SEARCH_DIRS}
    PATH_SUFFIXES ${TBB_PATH_SUFFIXES}
    )

  # On Unix, TBB sometimes uses linker scripts instead of symlinks, so parse the linker script
  # and correct the library name if so
  IF ( UNIX AND EXISTS ${Tbb_${COMPONENT}_LIBRARY} )
    # Ignore files where the first four bytes equals the ELF magic number
    FILE ( READ ${Tbb_${COMPONENT}_LIBRARY} Tbb_${COMPONENT}_HEX OFFSET 0 LIMIT 4 HEX )
    IF ( NOT ${Tbb_${COMPONENT}_HEX} STREQUAL "7f454c46" )
      # Read the first 1024 bytes of the library and match against an "INPUT (file)" regex
      FILE ( READ ${Tbb_${COMPONENT}_LIBRARY} Tbb_${COMPONENT}_ASCII OFFSET 0 LIMIT 1024 )
      IF ( "${Tbb_${COMPONENT}_ASCII}" MATCHES "INPUT \\(([^(]+)\\)")
        # Extract the directory and apply the matched text (in brackets)
        GET_FILENAME_COMPONENT ( Tbb_${COMPONENT}_DIR "${Tbb_${COMPONENT}_LIBRARY}" DIRECTORY )
        SET ( Tbb_${COMPONENT}_LIBRARY "${Tbb_${COMPONENT}_DIR}/${CMAKE_MATCH_1}" )
      ENDIF ()
    ENDIF ()
  ENDIF ()

  LIST ( APPEND Tbb_LIB_COMPONENTS ${Tbb_${COMPONENT}_LIBRARY} )

  IF ( Tbb_${COMPONENT}_LIBRARY )
    SET ( TBB_${COMPONENT}_FOUND TRUE )
  ELSE ()
    SET ( TBB_${COMPONENT}_FOUND FALSE )
  ENDIF ()
ENDFOREACH ()

# reset lib suffix

SET ( CMAKE_FIND_LIBRARY_SUFFIXES ${_TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

# ------------------------------------------------------------------------
#  Cache and set TBB_FOUND
# ------------------------------------------------------------------------

INCLUDE ( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( TBB
  FOUND_VAR TBB_FOUND
  REQUIRED_VARS
    Tbb_INCLUDE_DIR
    Tbb_LIB_COMPONENTS
  VERSION_VAR Tbb_VERSION
  HANDLE_COMPONENTS
)

IF ( TBB_FOUND )
  SET ( Tbb_LIBRARIES
    ${Tbb_LIB_COMPONENTS}
  )
  SET ( Tbb_INCLUDE_DIRS ${Tbb_INCLUDE_DIR} )
  SET ( Tbb_DEFINITIONS ${PC_Tbb_CFLAGS_OTHER} )

  SET ( Tbb_LIBRARY_DIRS "" )
  FOREACH ( LIB ${Tbb_LIB_COMPONENTS} )
    GET_FILENAME_COMPONENT ( _TBB_LIBDIR ${LIB} DIRECTORY )
    LIST ( APPEND Tbb_LIBRARY_DIRS ${_TBB_LIBDIR} )
  ENDFOREACH ()
  LIST ( REMOVE_DUPLICATES Tbb_LIBRARY_DIRS )

  # CMake sometimes struggles to follow TBB's imported target, as libtbb.so
  # is usually just a file which defines "INPUT (libtbb.so.2)". As it's not
  # a symlink, we need to include this directory manually incase it's not on
  # the environment library path

  LINK_DIRECTORIES ( ${Tbb_LIBRARY_DIRS} )

  # Configure imported targets

  FOREACH ( COMPONENT ${TBB_FIND_COMPONENTS} )
    IF ( NOT TARGET TBB::${COMPONENT} )
      ADD_LIBRARY ( TBB::${COMPONENT} UNKNOWN IMPORTED )
      SET_TARGET_PROPERTIES ( TBB::${COMPONENT} PROPERTIES
        IMPORTED_LOCATION "${Tbb_${COMPONENT}_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${Tbb_DEFINITIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${Tbb_INCLUDE_DIR}"
      )
    ENDIF ()
  ENDFOREACH ()
ELSEIF ( TBB_FIND_REQUIRED )
  MESSAGE ( FATAL_ERROR "Unable to find TBB")
ENDIF ()
