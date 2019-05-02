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

FindBlosc
---------

Find Blosc include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(Blosc
    [version] [EXACT]      # Minimum or EXACT version e.g. 1.5.0
    [REQUIRED]             # Fail with error if Blosc is not found
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``Blosc::blosc``
  This module defines IMPORTED target Blosc::Blosc, if Blosc has been found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Blosc_FOUND``
  True if the system has the Blosc library.
``Blosc_VERSION``
  The version of the Blosc library which was found.
``Blosc_INCLUDE_DIRS``
  Include directories needed to use Blosc.
``Blosc_LIBRARIES``
  Libraries needed to link to Blosc.
``Blosc_LIBRARY_DIRS``
  Blosc library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Blosc_INCLUDE_DIR``
  The directory containing ``blosc.h``.
``Blosc_LIBRARY``
  The path to the Blosc library.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``BLOSC_ROOT``
  Preferred installation prefix.
``BLOSC_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``BLOSC_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

MARK_AS_ADVANCED (
  Blosc_INCLUDE_DIR
  Blosc_LIBRARY
)

# Append BLOSC_ROOT or $ENV{BLOSC_ROOT} if set (prioritize the direct cmake var)
SET ( _BLOSC_ROOT_SEARCH_DIR "" )

IF ( BLOSC_ROOT )
  LIST ( APPEND _BLOSC_ROOT_SEARCH_DIR ${BLOSC_ROOT} )
ELSE ()
  SET ( _ENV_BLOSC_ROOT $ENV{BLOSC_ROOT} )
  IF ( _ENV_BLOSC_ROOT )
    LIST ( APPEND _BLOSC_ROOT_SEARCH_DIR ${_ENV_BLOSC_ROOT} )
  ENDIF ()
ENDIF ()

# Additionally try and use pkconfig to find blosc

FIND_PACKAGE ( PkgConfig )
PKG_CHECK_MODULES ( PC_Blosc QUIET blosc )

# ------------------------------------------------------------------------
#  Search for blosc include DIR
# ------------------------------------------------------------------------

SET ( _BLOSC_INCLUDE_SEARCH_DIRS "" )
LIST ( APPEND _BLOSC_INCLUDE_SEARCH_DIRS
  ${BLOSC_INCLUDEDIR}
  ${_BLOSC_ROOT_SEARCH_DIR}
  ${PC_Blosc_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Look for a standard blosc header file.
FIND_PATH ( Blosc_INCLUDE_DIR blosc.h
  NO_DEFAULT_PATH
  PATHS ${_BLOSC_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include
  )

IF ( EXISTS "${Blosc_INCLUDE_DIR}/blosc.h" )
  FILE ( STRINGS "${Blosc_INCLUDE_DIR}/blosc.h"
    _blosc_version_major_string REGEX "#define BLOSC_VERSION_MAJOR +[0-9]+ "
    )
  STRING ( REGEX REPLACE "#define BLOSC_VERSION_MAJOR +([0-9]+).*$" "\\1"
    _blosc_version_major_string "${_blosc_version_major_string}"
    )
  STRING ( STRIP "${_blosc_version_major_string}" Blosc_VERSION_MAJOR )

  FILE ( STRINGS "${Blosc_INCLUDE_DIR}/blosc.h"
     _blosc_version_minor_string REGEX "#define BLOSC_VERSION_MINOR +[0-9]+ "
    )
  STRING ( REGEX REPLACE "#define BLOSC_VERSION_MINOR +([0-9]+).*$" "\\1"
    _blosc_version_minor_string "${_blosc_version_minor_string}"
    )
  STRING ( STRIP "${_blosc_version_minor_string}" Blosc_VERSION_MINOR )

  UNSET ( _blosc_version_major_string )
  UNSET ( _blosc_version_minor_string )

  SET ( Blosc_VERSION ${Blosc_VERSION_MAJOR}.${Blosc_VERSION_MINOR} )
ENDIF ()

# ------------------------------------------------------------------------
#  Search for blosc lib DIR
# ------------------------------------------------------------------------

SET ( _BLOSC_LIBRARYDIR_SEARCH_DIRS "" )
LIST ( APPEND _BLOSC_LIBRARYDIR_SEARCH_DIRS
  ${BLOSC_LIBRARYDIR}
  ${_BLOSC_ROOT_SEARCH_DIR}
  ${PC_Blosc_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Static library setup
IF ( UNIX AND BLOSC_USE_STATIC_LIBS )
  SET ( _BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
  SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".a" )
ENDIF ()

SET ( BLOSC_PATH_SUFFIXES
  lib64
  lib
)

FIND_LIBRARY ( Blosc_LIBRARY blosc
  NO_DEFAULT_PATH
  PATHS ${_BLOSC_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${BLOSC_PATH_SUFFIXES}
)

IF ( UNIX AND BLOSC_USE_STATIC_LIBS )
  SET ( CMAKE_FIND_LIBRARY_SUFFIXES ${_BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES} )
  UNSET ( _BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES )
ENDIF ()

# ------------------------------------------------------------------------
#  Cache and set Blosc_FOUND
# ------------------------------------------------------------------------

INCLUDE ( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( Blosc
  FOUND_VAR Blosc_FOUND
  REQUIRED_VARS
    Blosc_LIBRARY
    Blosc_INCLUDE_DIR
  VERSION_VAR Blosc_VERSION
)

IF ( Blosc_FOUND )
  SET ( Blosc_LIBRARIES ${Blosc_LIBRARY} )
  SET ( Blosc_INCLUDE_DIRS ${Blosc_INCLUDE_DIR} )
  SET ( Blosc_DEFINITIONS ${PC_Blosc_CFLAGS_OTHER} )

  GET_FILENAME_COMPONENT ( Blosc_LIBRARY_DIRS ${Blosc_LIBRARY} DIRECTORY )

  IF ( NOT TARGET Blosc::blosc )
    ADD_LIBRARY ( Blosc::blosc UNKNOWN IMPORTED )
    SET_TARGET_PROPERTIES ( Blosc::blosc PROPERTIES
      IMPORTED_LOCATION "${Blosc_LIBRARIES}"
      INTERFACE_COMPILE_DEFINITIONS "${Blosc_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${Blosc_INCLUDE_DIRS}"
    )
  ENDIF ()
ELSEIF ( Blosc_FIND_REQUIRED )
  MESSAGE ( FATAL_ERROR "Unable to find Blosc")
ENDIF ()
