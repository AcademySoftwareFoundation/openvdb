# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
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

cmake_minimum_required(VERSION 3.12)
include(GNUInstallDirs)


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
if(USE_PKGCONFIG)
  if(NOT DEFINED PKG_CONFIG_FOUND)
    find_package(PkgConfig)
  endif()
  pkg_check_modules(PC_CppUnit QUIET cppunit)
endif()

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
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR} include
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
  if(CPPUNIT_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  else()
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "_dll.lib")
  endif()
else()
  if(CPPUNIT_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
endif()

# Build suffix directories

find_library(CppUnit_LIBRARY cppunit
  ${_FIND_CPPUNIT_ADDITIONAL_OPTIONS}
  PATHS ${_CPPUNIT_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
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
  # Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
  # lib is in use. If win32, we can't mark the import .libs as shared, so
  # these are always marked as UNKNOWN. Otherwise, infer from extension.
  set(CPPUNIT_LIB_TYPE UNKNOWN)
  if(CPPUNIT_USE_STATIC_LIBS)
    set(CPPUNIT_LIB_TYPE STATIC)
  elseif(UNIX)
    get_filename_component(_CPPUNIT_EXT ${CppUnit_LIBRARY} EXT)
    if(_CPPUNIT_EXT STREQUAL ".a")
      set(CPPUNIT_LIB_TYPE STATIC)
    elseif(_CPPUNIT_EXT STREQUAL ".so" OR
           _CPPUNIT_EXT STREQUAL ".dylib")
      set(CPPUNIT_LIB_TYPE SHARED)
    endif()
  endif()

  set(CppUnit_LIBRARIES ${CppUnit_LIBRARY})
  set(CppUnit_INCLUDE_DIRS ${CppUnit_INCLUDE_DIR})

  get_filename_component(CppUnit_LIBRARY_DIRS ${CppUnit_LIBRARY} DIRECTORY)

  if(NOT TARGET CppUnit::cppunit)
    add_library(CppUnit::cppunit ${CPPUNIT_LIB_TYPE} IMPORTED)
    set_target_properties(CppUnit::cppunit PROPERTIES
      IMPORTED_LOCATION "${CppUnit_LIBRARIES}"
      INTERFACE_COMPILE_OPTIONS "${PC_CppUnit_CFLAGS_OTHER}"
      INTERFACE_INCLUDE_DIRECTORIES "${CppUnit_INCLUDE_DIRS}"
    )
  endif()
elseif(CppUnit_FIND_REQUIRED)
  message(FATAL_ERROR "Unable to find CppUnit")
endif()
