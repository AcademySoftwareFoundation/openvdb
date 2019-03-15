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

FindCPPUnit
---------

Find CPPUnit include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(CPPUnit
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if CPPUnit is not found
    )


IMPORTED Targets
^^^^^^^^^^^^^^^^

``CPPUnit::CPPUnit``
  This module defines IMPORTED target CPPUnit::CPPUnit, if CPPUnit has been
  found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``CPPUnit_FOUND``
  True if the system has the CPPUnit library.
``CPPUnit_VERSION``
  The version of the CPPUnit library which was found.
``CPPUnit_INCLUDE_DIRS``
  Include directories needed to use CPPUnit.
``CPPUnit_LIBRARIES``
  Libraries needed to link to CPPUnit.
``CPPUnit_LIBRARY_DIRS``
  CPPUnit library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``CPPUnit_INCLUDE_DIR``
  The directory containing ``cppunit/config-auto.h``.
``CPPUnit_LIBRARY``
  The path to the CPPUnit library.

Hints
^^^^^^^^^^^^^^^

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
  CPPUnit_INCLUDE_DIR
  CPPUnit_LIBRARY
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

# Additionally try and use pkconfig to find ILMBase

FIND_PACKAGE ( PkgConfig )
PKG_CHECK_MODULES ( PC_CPPUnit QUIET cppunit )

# ------------------------------------------------------------------------
#  Search for CPPUnit include DIR
# ------------------------------------------------------------------------

SET ( _CPPUNIT_INCLUDE_SEARCH_DIRS "" )
LIST ( APPEND _CPPUNIT_INCLUDE_SEARCH_DIRS
  ${CPPUNIT_INCLUDEDIR}
  ${_CPPUNIT_ROOT_SEARCH_DIR}
  ${PC_CPPUnit_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

# Look for a standard cppunit header file.
FIND_PATH ( CPPUnit_INCLUDE_DIR cppunit/config-auto.h
  NO_DEFAULT_PATH
  PATHS ${_CPPUNIT_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include
  )

IF ( EXISTS "${CPPUnit_INCLUDE_DIR}/cppunit/config-auto.h" )
  FILE ( STRINGS "${CPPUnit_INCLUDE_DIR}/cppunit/config-auto.h"
    _cppunit_version_string REGEX "#define CPPUNIT_VERSION "
    )
  STRING ( REGEX REPLACE "#define CPPUNIT_VERSION +\"(.+)\".*$" "\\1"
    _cppunit_version_string "${_cppunit_version_string}"
    )
  STRING ( STRIP "${_cppunit_version_string}" CPPUnit_VERSION )
  UNSET ( _cppunit_version_string )
ENDIF ()

# ------------------------------------------------------------------------
#  Search for CPPUnit lib DIR
# ------------------------------------------------------------------------

SET ( _CPPUNIT_LIBRARYDIR_SEARCH_DIRS "" )
LIST ( APPEND _CPPUNIT_LIBRARYDIR_SEARCH_DIRS
  ${CPPUNIT_LIBRARYDIR}
  ${_CPPUNIT_ROOT_SEARCH_DIR}
  ${PC_CPPUnit_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
  )

SET ( _CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} )

IF ( CPPUNIT_USE_STATIC_LIBS )
  IF ( UNIX )
    SET ( CMAKE_FIND_LIBRARY_SUFFIXES ".a" )
  ENDIF ()
ENDIF ()

# Build suffix directories

SET ( CPPUNIT_PATH_SUFFIXES
  lib64
  lib
)

FIND_LIBRARY ( CPPUnit_LIBRARY cppunit
  NO_DEFAULT_PATH
  PATHS ${_CPPUNIT_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${CPPUNIT_PATH_SUFFIXES}
  )

SET ( CMAKE_FIND_LIBRARY_SUFFIXES ${_CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

# ------------------------------------------------------------------------
#  Cache and set CPPUnit_FOUND
# ------------------------------------------------------------------------

INCLUDE ( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( CPPUnit
  FOUND_VAR CPPUnit_FOUND
  REQUIRED_VARS
    CPPUnit_LIBRARY
    CPPUnit_INCLUDE_DIR
  VERSION_VAR CPPUnit_VERSION
)

IF ( CPPUnit_FOUND )
  SET ( CPPUnit_LIBRARIES ${CPPUnit_LIBRARY} )
  SET ( CPPUnit_INCLUDE_DIRS ${CPPUnit_INCLUDE_DIR} )
  SET ( CPPUnit_DEFINITIONS ${PC_CPPUnit_CFLAGS_OTHER} )

  GET_FILENAME_COMPONENT ( CPPUnit_LIBRARY_DIRS ${CPPUnit_LIBRARY} DIRECTORY )

  IF ( NOT TARGET CPPUnit::CPPUnit )
    ADD_LIBRARY ( CPPUnit::CPPUnit UNKNOWN IMPORTED )
    SET_TARGET_PROPERTIES ( CPPUnit::CPPUnit PROPERTIES
      IMPORTED_LOCATION "${CPPUnit_LIBRARIES}"
      INTERFACE_COMPILE_DEFINITIONS "${CPPUnit_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${CPPUnit_INCLUDE_DIRS}"
    )
  ENDIF ()
ELSEIF ( CPPUnit_FIND_REQUIRED )
  MESSAGE ( FATAL_ERROR "Unable to find CPPUnit")
ENDIF ()
