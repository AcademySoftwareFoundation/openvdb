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
``CppUnit_RELEASE_LIBRARIES``
  Libraries needed to link to the release version of CppUnit.
``CppUnit_RELEASE_LIBRARY_DIRS``
  CppUnit release library directories.
``CppUnit_DEBUG_LIBRARIES``
  Libraries needed to link to the debug version of CppUnit.
``CppUnit_DEBUG_LIBRARY_DIRS``
  CppUnit debug library directories.

Deprecated - use [RELEASE|DEBUG] variants:

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
``CPPUNIT_DEBUG_SUFFIX``
  Suffix of the debug version of CppUnit libs. Defaults to "_d".
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``CPPUNIT_USE_STATIC_LIBS``
  Only search for static cppunit libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
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

if(NOT DEFINED CPPUNIT_DEBUG_SUFFIX)
  set(CPPUNIT_DEBUG_SUFFIX "d")
endif()

set(_CPPUNIT_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _CPPUNIT_LIBRARYDIR_SEARCH_DIRS
  ${CPPUNIT_LIBRARYDIR}
  ${_CPPUNIT_ROOT}
  ${PC_CppUnit_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Library suffix handling

set(_CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(MSVC)
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

list(APPEND CPPUNIT_BUILD_TYPES RELEASE DEBUG)

foreach(BUILD_TYPE ${CPPUNIT_BUILD_TYPES})
  set(_CPPUNIT_LIB_NAME cppunit)
  if(BUILD_TYPE STREQUAL DEBUG)
    set(_CPPUNIT_LIB_NAME "${_CPPUNIT_LIB_NAME}${CPPUNIT_DEBUG_SUFFIX}")
  endif()

  # Find the lib
  find_library(CppUnit_LIBRARY_${BUILD_TYPE} ${_CPPUNIT_LIB_NAME}
    ${_FIND_CPPUNIT_ADDITIONAL_OPTIONS}
    PATHS ${_CPPUNIT_LIBRARYDIR_SEARCH_DIRS}
    PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
  )
endforeach()

# Reset library suffix

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_CPPUNIT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

if(CppUnit_LIBRARY_DEBUG AND CppUnit_LIBRARY_RELEASE)
  # if the generator is multi-config or if CMAKE_BUILD_TYPE is set for
  # single-config generators, set optimized and debug libraries
  get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_isMultiConfig OR CMAKE_BUILD_TYPE)
    set(CppUnit_LIBRARY optimized ${CppUnit_LIBRARY_RELEASE} debug ${CppUnit_LIBRARY_DEBUG})
  else()
    # For single-config generators where CMAKE_BUILD_TYPE has no value,
    # just use the release libraries
    set(CppUnit_LIBRARY ${CppUnit_LIBRARY_RELEASE})
  endif()
  # FIXME: This probably should be set for both cases
  set(CppUnit_LIBRARIES optimized ${CppUnit_LIBRARY_RELEASE} debug ${CppUnit_LIBRARY_DEBUG})
endif()

# if only the release version was found, set the debug variable also to the release version
if(CppUnit_LIBRARY_RELEASE AND NOT CppUnit_LIBRARY_DEBUG)
  set(CppUnit_LIBRARY_DEBUG ${CppUnit_LIBRARY_RELEASE})
  set(CppUnit_LIBRARY       ${CppUnit_LIBRARY_RELEASE})
  set(CppUnit_LIBRARIES     ${CppUnit_LIBRARY_RELEASE})
endif()

# if only the debug version was found, set the release variable also to the debug version
if(CppUnit_LIBRARY_DEBUG AND NOT CppUnit_LIBRARY_RELEASE)
  set(CppUnit_LIBRARY_RELEASE ${CppUnit_LIBRARY_DEBUG})
  set(CppUnit_LIBRARY         ${CppUnit_LIBRARY_DEBUG})
  set(CppUnit_LIBRARIES       ${CppUnit_LIBRARY_DEBUG})
endif()

# If the debug & release library ends up being the same, omit the keywords
if("${CppUnit_LIBRARY_RELEASE}" STREQUAL "${CppUnit_LIBRARY_DEBUG}")
  set(CppUnit_LIBRARY   ${CppUnit_LIBRARY_RELEASE} )
  set(CppUnit_LIBRARIES ${CppUnit_LIBRARY_RELEASE} )
endif()

if(CppUnit_LIBRARY)
  set(CppUnit_FOUND TRUE)
else()
  set(CppUnit_FOUND FALSE)
endif()

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

if(NOT CppUnit_FOUND)
  if(CppUnit_FIND_REQUIRED)
    message(FATAL_ERROR "Unable to find CppUnit")
  endif()
  return()
endif()

# Partition release/debug lib vars

set(CppUnit_RELEASE_LIBRARIES ${CppUnit_LIBRARY_RELEASE})
get_filename_component(CppUnit_RELEASE_LIBRARY_DIRS ${CppUnit_RELEASE_LIBRARIES} DIRECTORY)
set(CppUnit_DEBUG_LIBRARIES ${CppUnit_LIBRARY_DEBUG})
get_filename_component(CppUnit_DEBUG_LIBRARY_DIRS ${CppUnit_DEBUG_LIBRARIES} DIRECTORY)
set(CppUnit_LIBRARIES ${CppUnit_RELEASE_LIBRARIES})
set(CppUnit_LIBRARY_DIRS ${CppUnit_RELEASE_LIBRARY_DIRS})
set(CppUnit_INCLUDE_DIRS ${CppUnit_INCLUDE_DIR})

# Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
# lib is in use. If win32, we can't mark the import .libs as shared, so
# these are always marked as UNKNOWN. Otherwise, infer from extension.
set(CPPUNIT_LIB_TYPE UNKNOWN)
if(CPPUNIT_USE_STATIC_LIBS)
  set(CPPUNIT_LIB_TYPE STATIC)
elseif(UNIX)
  get_filename_component(_CPPUNIT_EXT ${CppUnit_LIBRARY_RELEASE} EXT)
  if(_CPPUNIT_EXT STREQUAL ".a")
    set(CPPUNIT_LIB_TYPE STATIC)
  elseif(_CPPUNIT_EXT STREQUAL ".so" OR
         _CPPUNIT_EXT STREQUAL ".dylib")
    set(CPPUNIT_LIB_TYPE SHARED)
  endif()
endif()

if(NOT TARGET CppUnit::cppunit)
  add_library(CppUnit::cppunit ${CPPUNIT_LIB_TYPE} IMPORTED)
  set_target_properties(CppUnit::cppunit PROPERTIES
    INTERFACE_COMPILE_OPTIONS "${PC_CppUnit_CFLAGS_OTHER}"
    INTERFACE_INCLUDE_DIRECTORIES "${CppUnit_INCLUDE_DIRS}")

  # Standard location
  set_target_properties(CppUnit::cppunit PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${CppUnit_LIBRARY}")

  # Release location
  if(EXISTS "${CppUnit_LIBRARY_RELEASE}")
    set_property(TARGET CppUnit::cppunit APPEND PROPERTY
      IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(CppUnit::cppunit PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
      IMPORTED_LOCATION_RELEASE "${CppUnit_LIBRARY_RELEASE}")
  endif()

  # Debug location
  if(EXISTS "${CppUnit_LIBRARY_DEBUG}")
    set_property(TARGET CppUnit::cppunit APPEND PROPERTY
      IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(CppUnit::cppunit PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
      IMPORTED_LOCATION_DEBUG "${CppUnit_LIBRARY_DEBUG}")
  endif()
endif()
