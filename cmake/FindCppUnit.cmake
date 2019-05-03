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
  The directory containing ``cppunit/Portability.h``.
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

mark_as_advanced(
  CppUnit_INCLUDE_DIR
  CppUnit_LIBRARY
)

# Append CPPUNIT_ROOT or $ENV{CPPUNIT_ROOT} if set (prioritize the direct cmake var)
set(_CPPUNIT_ROOT_SEARCH_DIR "")

if(CPPUNIT_ROOT)
  list(APPEND _CPPUNIT_ROOT_SEARCH_DIR ${CPPUNIT_ROOT})
else()
  set(_ENV_CPPUNIT_ROOT $ENV{CPPUNIT_ROOT})
  if(_ENV_CPPUNIT_ROOT)
    list(APPEND _CPPUNIT_ROOT_SEARCH_DIR ${_ENV_CPPUNIT_ROOT})
  endif()
endif()

# Additionally try and use pkconfig to find cppunit

find_package(PkgConfig)
pkg_check_modules(PC_CppUnit QUIET cppunit)

# ------------------------------------------------------------------------
#  Search for CppUnit include DIR
# ------------------------------------------------------------------------

set(_CPPUNIT_INCLUDE_SEARCH_DIRS "")
list(APPEND _CPPUNIT_INCLUDE_SEARCH_DIRS
  ${CPPUNIT_INCLUDEDIR}
  ${_CPPUNIT_ROOT_SEARCH_DIR}
  ${PC_CppUnit_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard cppunit header file.
find_path(CppUnit_INCLUDE_DIR cppunit/Portability.h
  NO_DEFAULT_PATH
  PATHS ${_CPPUNIT_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include
)

if(EXISTS "${CppUnit_INCLUDE_DIR}/cppunit/Portability.h")
  file(STRINGS "${CppUnit_INCLUDE_DIR}/cppunit/Portability.h"
    _cppunit_version_string REGEX "#define CPPUNIT_VERSION "
  )
  string(REGEX REPLACE "#define CPPUNIT_VERSION +\"(.+)\".*$" "\\1"
    _cppunit_version_string "${_cppunit_version_string}"
  )
  string(STRIP "${_cppunit_version_string}" CppUnit_VERSION)
  unset(_cppunit_version_string )
endif()

# ------------------------------------------------------------------------
#  Search for CppUnit lib DIR
# ------------------------------------------------------------------------

set(_CPPUNIT_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _CPPUNIT_LIBRARYDIR_SEARCH_DIRS
  ${CPPUNIT_LIBRARYDIR}
  ${_CPPUNIT_ROOT_SEARCH_DIR}
  ${PC_CppUnit_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Library suffix handling

set(_CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(WIN32)
  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES
    "_dll.lib"
  )
elseif(UNIX)
  if(CPPUNIT_USE_STATIC_LIBS)
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES
      ".a"
    )
  endif()
endif()

# Build suffix directories

set(CPPUNIT_PATH_SUFFIXES
  lib64
  lib
)

find_library(CppUnit_LIBRARY cppunit
  NO_DEFAULT_PATH
  PATHS ${_CPPUNIT_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${CPPUNIT_PATH_SUFFIXES}
)

# Reset library suffix

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

# ------------------------------------------------------------------------
#  Cache and set CppUnit_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CppUnit
  FOUND_VAR CppUnit_FOUND
  REQUIRED_VARS
    CppUnit_LIBRARY
    CppUnit_INCLUDE_DIR
  VERSION_VAR CppUnit_VERSION
)

if(CppUnit_FOUND)
  set(CppUnit_LIBRARIES ${CppUnit_LIBRARY})
  set(CppUnit_INCLUDE_DIRS ${CppUnit_INCLUDE_DIR})
  set(CppUnit_DEFINITIONS ${PC_CppUnit_CFLAGS_OTHER})

  get_filename_component(CppUnit_LIBRARY_DIRS ${CppUnit_LIBRARY} DIRECTORY)

  if(NOT TARGET CppUnit::cppunit)
    add_library(CppUnit::cppunit UNKNOWN IMPORTED)
    set_target_properties(CppUnit::cppunit PROPERTIES
      IMPORTED_LOCATION "${CppUnit_LIBRARIES}"
      INTERFACE_COMPILE_DEFINITIONS "${CppUnit_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${CppUnit_INCLUDE_DIRS}"
    )
  endif()
elseif(CppUnit_FIND_REQUIRED)
  message(FATAL_ERROR "Unable to find CppUnit")
endif()
