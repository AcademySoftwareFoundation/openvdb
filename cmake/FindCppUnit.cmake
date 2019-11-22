# Copyright (c) DreamWorks Animation LLC
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

``CppUnit_ROOT``
  Preferred installation prefix.
``CPPUNIT_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``CPPUNIT_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``CPPUNIT_USE_STATIC_LIBS``
  Only search for static cppunit libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.3)

# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

mark_as_advanced(
  CppUnit_INCLUDE_DIR
  CppUnit_LIBRARY
)

set(_FIND_CPPUNIT_ADDITIONAL_OPTIONS "")
if(DISABLE_CMAKE_SEARCH_PATHS)
  set(_FIND_CPPUNIT_ADDITIONAL_OPTIONS NO_DEFAULT_PATH)
endif()

# Set _CPPUNIT_ROOT based on a user provided root var. Xxx_ROOT and ENV{Xxx_ROOT}
# are prioritised over the legacy capitalized XXX_ROOT variables for matching
# CMake 3.12 behaviour
# @todo  deprecate -D and ENV CPPUNIT_ROOT from CMake 3.12
if(CppUnit_ROOT)
  set(_CPPUNIT_ROOT ${CppUnit_ROOT})
elseif(DEFINED ENV{CppUnit_ROOT})
  set(_CPPUNIT_ROOT $ENV{CppUnit_ROOT})
elseif(CPPUNIT_ROOT)
  set(_CPPUNIT_ROOT ${CPPUNIT_ROOT})
elseif(DEFINED ENV{CPPUNIT_ROOT})
  set(_CPPUNIT_ROOT $ENV{CPPUNIT_ROOT})
endif()

# Additionally try and use pkconfig to find cppunit

if(NOT DEFINED PKG_CONFIG_FOUND)
  find_package(PkgConfig)
endif()
pkg_check_modules(PC_CppUnit QUIET cppunit)

# ------------------------------------------------------------------------
#  Search for CppUnit include DIR
# ------------------------------------------------------------------------

set(_CPPUNIT_INCLUDE_SEARCH_DIRS "")
list(APPEND _CPPUNIT_INCLUDE_SEARCH_DIRS
  ${CPPUNIT_INCLUDEDIR}
  ${_CPPUNIT_ROOT}
  ${PC_CppUnit_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard cppunit header file.
find_path(CppUnit_INCLUDE_DIR cppunit/Portability.h
  ${_FIND_CPPUNIT_ADDITIONAL_OPTIONS}
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
  ${_CPPUNIT_ROOT}
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
  ${_FIND_CPPUNIT_ADDITIONAL_OPTIONS}
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
