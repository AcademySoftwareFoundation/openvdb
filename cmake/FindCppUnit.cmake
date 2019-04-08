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

FindCppUnit
-----------

Find CppUnit include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(CppUnit
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if CppUnit is not found
    )


IMPORTED Targets
^^^^^^^^^^^^^^^^

``CppUnit::cppunit``
  This module defines IMPORTED target CppUnit::cppunit, if CppUnit has been
  found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``CppUnit_FOUND``
  True if the system has the CppUnit library.
``CppUnit_VERSION``
  The version of the CppUnit library which was found.
``CppUnit_INCLUDE_DIRS``
  Include directories needed to use CppUnit.
``CppUnit_LIBRARIES``
  Libraries needed to link to CppUnit.
``CppUnit_LIBRARY_DIRS``
  CppUnit library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``CppUnit_INCLUDE_DIR``
  The directory containing ``cppunit/config-auto.h``.
``CppUnit_LIBRARY``
  The path to the CppUnit library.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``CPPUNIT_ROOT``
  Preferred installation prefix.
``CPPUNIT_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``CPPUNIT_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

MARK_AS_ADVANCED (
  CppUnit_INCLUDE_DIR
  CppUnit_LIBRARY
)

# Append CPPUNIT_ROOT or $ENV{CPPUNIT_ROOT} if set (prioritize the direct cmake var)
SET ( _CPPUNIT_ROOT_SEARCH_DIR "" )

IF ( CPPUNIT_ROOT )
  LIST ( APPEND _CPPUNIT_ROOT_SEARCH_DIR ${CPPUNIT_ROOT} )
ELSE ()
  SET ( _ENV_CPPUNIT_ROOT $ENV{CPPUNIT_ROOT} )
  IF ( _ENV_CPPUNIT_ROOT )
    LIST ( APPEND _CPPUNIT_ROOT_SEARCH_DIR ${_ENV_CPPUNIT_ROOT} )
  ENDIF ()
ENDIF ()

# Additionally try and use pkconfig to find cppunit

FIND_PACKAGE ( PkgConfig )
PKG_CHECK_MODULES ( PC_CppUnit QUIET cppunit )

# ------------------------------------------------------------------------
#  Search for CppUnit include DIR
# ------------------------------------------------------------------------

SET ( _CPPUNIT_INCLUDE_SEARCH_DIRS "" )
LIST ( APPEND _CPPUNIT_INCLUDE_SEARCH_DIRS
  ${CPPUNIT_INCLUDEDIR}
  ${_CPPUNIT_ROOT_SEARCH_DIR}
  ${PC_CppUnit_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Look for a standard cppunit header file.
FIND_PATH ( CppUnit_INCLUDE_DIR cppunit/config-auto.h
  NO_DEFAULT_PATH
  PATHS ${_CPPUNIT_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include
  )

IF ( EXISTS "${CppUnit_INCLUDE_DIR}/cppunit/config-auto.h" )
  FILE ( STRINGS "${CppUnit_INCLUDE_DIR}/cppunit/config-auto.h"
    _cppunit_version_string REGEX "#define CPPUNIT_VERSION "
    )
  STRING ( REGEX REPLACE "#define CPPUNIT_VERSION +\"(.+)\".*$" "\\1"
    _cppunit_version_string "${_cppunit_version_string}"
    )
  STRING ( STRIP "${_cppunit_version_string}" CppUnit_VERSION )
  UNSET ( _cppunit_version_string )
ENDIF ()

# ------------------------------------------------------------------------
#  Search for CppUnit lib DIR
# ------------------------------------------------------------------------

SET ( _CPPUNIT_LIBRARYDIR_SEARCH_DIRS "" )
LIST ( APPEND _CPPUNIT_LIBRARYDIR_SEARCH_DIRS
  ${CPPUNIT_LIBRARYDIR}
  ${_CPPUNIT_ROOT_SEARCH_DIR}
  ${PC_CppUnit_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Static library setup
IF ( UNIX AND CPPUNIT_USE_STATIC_LIBS )
  SET ( _CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )
  SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".a" )
ENDIF ()

# Build suffix directories

SET ( CPPUNIT_PATH_SUFFIXES
  lib64
  lib
)

FIND_LIBRARY ( CppUnit_LIBRARY cppunit
  NO_DEFAULT_PATH
  PATHS ${_CPPUNIT_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${CPPUNIT_PATH_SUFFIXES}
  )

IF ( UNIX AND CPPUNIT_USE_STATIC_LIBS )
  SET ( CMAKE_FIND_LIBRARY_SUFFIXES ${_CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES} )
  UNSET ( _CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES )
ENDIF ()

# ------------------------------------------------------------------------
#  Cache and set CppUnit_FOUND
# ------------------------------------------------------------------------

INCLUDE ( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( CppUnit
  FOUND_VAR CppUnit_FOUND
  REQUIRED_VARS
    CppUnit_LIBRARY
    CppUnit_INCLUDE_DIR
  VERSION_VAR CppUnit_VERSION
)

IF ( CppUnit_FOUND )
  SET ( CppUnit_LIBRARIES ${CppUnit_LIBRARY} )
  SET ( CppUnit_INCLUDE_DIRS ${CppUnit_INCLUDE_DIR} )
  SET ( CppUnit_DEFINITIONS ${PC_CppUnit_CFLAGS_OTHER} )

  GET_FILENAME_COMPONENT ( CppUnit_LIBRARY_DIRS ${CppUnit_LIBRARY} DIRECTORY )

  IF ( NOT TARGET CppUnit::cppunit )
    ADD_LIBRARY ( CppUnit::cppunit UNKNOWN IMPORTED )
    SET_TARGET_PROPERTIES ( CppUnit::cppunit PROPERTIES
      IMPORTED_LOCATION "${CppUnit_LIBRARIES}"
      INTERFACE_COMPILE_DEFINITIONS "${CppUnit_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${CppUnit_INCLUDE_DIRS}"
    )
  ENDIF ()
ELSEIF ( CppUnit_FIND_REQUIRED )
  MESSAGE ( FATAL_ERROR "Unable to find CppUnit")
ENDIF ()
