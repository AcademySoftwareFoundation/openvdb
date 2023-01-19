# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[.rst:

FindLog4cplus
-------------

Find Log4cplus include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(Log4cplus
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if Log4cplus is not found
    )


IMPORTED Targets
^^^^^^^^^^^^^^^^

``Log4cplus::Log4cplus``
  This module defines IMPORTED target Log4cplus::log4cplus, if Log4cplus has been
  found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Log4cplus_FOUND``
  True if the system has the Log4cplus library.
``Log4cplus_VERSION``
  The version of the Log4cplus library which was found.
``Log4cplus_INCLUDE_DIRS``
  Include directories needed to use Log4cplus.
``Log4cplus_RELEASE_LIBRARIES``
  Libraries needed to link to the release version of Log4cplus.
``Log4cplus_RELEASE_LIBRARY_DIRS``
  Log4cplus release library directories.
``Log4cplus_DEBUG_LIBRARIES``
  Libraries needed to link to the debug version of Log4cplus.
``Log4cplus_DEBUG_LIBRARY_DIRS``
  Log4cplus debug library directories.

Deprecated - use [RELEASE|DEBUG] variants:

``Log4cplus_LIBRARIES``
  Libraries needed to link to Log4cplus.
``Log4cplus_LIBRARY_DIRS``
  Log4cplus library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Log4cplus_INCLUDE_DIR``
  The directory containing ``log4cplus/version.h``.
``Log4cplus_LIBRARY``
  The path to the Log4cplus library. may include target_link_libraries() debug/optimized keywords.
``Log4cplus_LIBRARY_RELEASE``
  The path to the Log4cplus release library.
``Log4cplus_LIBRARY_DEBUG``
  The path to the Log4cplus debug library.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``Log4cplus_ROOT``
  Preferred installation prefix.
``LOG4CPLUS_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``LOG4CPLUS_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``LOG4CPLUS_DEBUG_SUFFIX``
  Suffix of the debug version of the log4cplus lib. Defaults to "D".
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``LOG4CPLUS_USE_STATIC_LIBS``
  Only search for static log4cplus libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
include(GNUInstallDirs)


mark_as_advanced(
  Log4cplus_INCLUDE_DIR
  Log4cplus_LIBRARY
)

set(_FIND_LOG4CPLUS_ADDITIONAL_OPTIONS "")
if(DISABLE_CMAKE_SEARCH_PATHS)
  set(_FIND_LOG4CPLUS_ADDITIONAL_OPTIONS NO_DEFAULT_PATH)
endif()

# Set _LOG4CPLUS_ROOT based on a user provided root var. Xxx_ROOT and ENV{Xxx_ROOT}
# are prioritised over the legacy capitalized XXX_ROOT variables for matching
# CMake 3.12 behaviour
# @todo  deprecate -D and ENV LOG4CPLUS_ROOT from CMake 3.12
if(Log4cplus_ROOT)
  set(_LOG4CPLUS_ROOT ${Log4cplus_ROOT})
elseif(DEFINED ENV{Log4cplus_ROOT})
  set(_LOG4CPLUS_ROOT $ENV{Log4cplus_ROOT})
elseif(LOG4CPLUS_ROOT)
  set(_LOG4CPLUS_ROOT ${LOG4CPLUS_ROOT})
elseif(DEFINED ENV{LOG4CPLUS_ROOT})
  set(_LOG4CPLUS_ROOT $ENV{LOG4CPLUS_ROOT})
endif()

# Additionally try and use pkconfig to find log4cplus
if(USE_PKGCONFIG)
  if(NOT DEFINED PKG_CONFIG_FOUND)
    find_package(PkgConfig)
  endif()
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_Log4cplus QUIET log4cplus)
  endif()
endif()

# ------------------------------------------------------------------------
#  Search for Log4cplus include DIR
# ------------------------------------------------------------------------

set(_LOG4CPLUS_INCLUDE_SEARCH_DIRS "")
list(APPEND _LOG4CPLUS_INCLUDE_SEARCH_DIRS
  ${LOG4CPLUS_INCLUDEDIR}
  ${_LOG4CPLUS_ROOT}
  ${PC_Log4cplus_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard log4cplus header file.
find_path(Log4cplus_INCLUDE_DIR log4cplus/version.h
  ${_FIND_LOG4CPLUS_ADDITIONAL_OPTIONS}
  PATHS ${_LOG4CPLUS_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR} include
)

if(EXISTS "${Log4cplus_INCLUDE_DIR}/log4cplus/version.h")
  file(STRINGS "${Log4cplus_INCLUDE_DIR}/log4cplus/version.h"
    _log4cplus_version_string REGEX "#define LOG4CPLUS_VERSION LOG4CPLUS_MAKE_VERSION"
  )
  string(REGEX REPLACE "#define LOG4CPLUS_VERSION LOG4CPLUS_MAKE_VERSION\((.*)\).*$" "\\1"
    _log4cplus_version_string "${_log4cplus_version_string}"
  )
  string(REGEX REPLACE "[(]([0-9]+),.*[)].*$" "\\1"
    Log4cplus_MAJOR_VERSION "${_log4cplus_version_string}"
  )
  string(REGEX REPLACE "[(].+, ([0-9]+),.+[)].*$" "\\1"
    Log4cplus_MINOR_VERSION "${_log4cplus_version_string}"
  )
  string(REGEX REPLACE "[(].*,.*, ([0-9]+)[)].*$" "\\1"
    Log4cplus_PATCH_VERSION "${_log4cplus_version_string}"
  )
  unset(_log4cplus_version_string)

  set(Log4cplus_VERSION ${Log4cplus_MAJOR_VERSION}.${Log4cplus_MINOR_VERSION}.${Log4cplus_PATCH_VERSION})
endif()

# ------------------------------------------------------------------------
#  Search for Log4cplus lib DIR
# ------------------------------------------------------------------------

if(NOT DEFINED LOG4CPLUS_DEBUG_SUFFIX)
  set(LOG4CPLUS_DEBUG_SUFFIX D)
endif()

set(_LOG4CPLUS_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _LOG4CPLUS_LIBRARYDIR_SEARCH_DIRS
  ${LOG4CPLUS_LIBRARYDIR}
  ${_LOG4CPLUS_ROOT}
  ${PC_Log4cplus_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Library suffix handling

set(_LOG4CPLUS_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(MSVC)
  if(LOG4CPLUS_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  endif()
else()
  if(LOG4CPLUS_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
endif()

set(Log4cplus_LIB_COMPONENTS "")
list(APPEND LOG4CPLUS_BUILD_TYPES RELEASE DEBUG)

foreach(BUILD_TYPE ${LOG4CPLUS_BUILD_TYPES})
  set(_LOG4CPLUS_LIB_NAME log4cplus)
  if(BUILD_TYPE STREQUAL DEBUG)
    set(_LOG4CPLUS_LIB_NAME "${_LOG4CPLUS_LIB_NAME}${LOG4CPLUS_DEBUG_SUFFIX}")
  endif()

  find_library(Log4cplus_LIBRARY_${BUILD_TYPE} ${_LOG4CPLUS_LIB_NAME}
    ${_FIND_LOG4CPLUS_ADDITIONAL_OPTIONS}
    PATHS ${_LOG4CPLUS_LIBRARYDIR_SEARCH_DIRS}
    PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
  )

  list(APPEND Log4cplus_LIB_COMPONENTS ${Log4cplus_LIBRARY_${BUILD_TYPE}})
endforeach()

# Reset library suffix

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_LOG4CPLUS_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_LOG4CPLUS_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

if(Log4cplus_LIBRARY_DEBUG AND Log4cplus_LIBRARY_RELEASE)
  # if the generator is multi-config or if CMAKE_BUILD_TYPE is set for
  # single-config generators, set optimized and debug libraries
  get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_isMultiConfig OR CMAKE_BUILD_TYPE)
    set(Log4cplus_LIBRARY optimized ${Log4cplus_LIBRARY_RELEASE} debug ${Log4cplus_LIBRARY_DEBUG})
  else()
    # For single-config generators where CMAKE_BUILD_TYPE has no value,
    # just use the release libraries
    set(Log4cplus_LIBRARY ${Log4cplus_LIBRARY_RELEASE})
  endif()
  # FIXME: This probably should be set for both cases
  set(Log4cplus_LIBRARIES optimized ${Log4cplus_LIBRARY_RELEASE} debug ${Log4cplus_LIBRARY_DEBUG})
endif()

# if only the release version was found, set the debug variable also to the release version
if(Log4cplus_LIBRARY_RELEASE AND NOT Log4cplus_LIBRARY_DEBUG)
  set(Log4cplus_LIBRARY_DEBUG ${Log4cplus_LIBRARY_RELEASE})
  set(Log4cplus_LIBRARY       ${Log4cplus_LIBRARY_RELEASE})
  set(Log4cplus_LIBRARIES     ${Log4cplus_LIBRARY_RELEASE})
endif()

# if only the debug version was found, set the release variable also to the debug version
if(Log4cplus_LIBRARY_DEBUG AND NOT Log4cplus_LIBRARY_RELEASE)
  set(Log4cplus_LIBRARY_RELEASE ${Log4cplus_LIBRARY_DEBUG})
  set(Log4cplus_LIBRARY         ${Log4cplus_LIBRARY_DEBUG})
  set(Log4cplus_LIBRARIES       ${Log4cplus_LIBRARY_DEBUG})
endif()

# If the debug & release library ends up being the same, omit the keywords
if("${Log4cplus_LIBRARY_RELEASE}" STREQUAL "${Log4cplus_LIBRARY_DEBUG}")
  set(Log4cplus_LIBRARY   ${Log4cplus_LIBRARY_RELEASE} )
  set(Log4cplus_LIBRARIES ${Log4cplus_LIBRARY_RELEASE} )
endif()

if(Log4cplus_LIBRARY)
  set(Log4cplus_FOUND TRUE)
else()
  set(Log4cplus_FOUND FALSE)
endif()

# ------------------------------------------------------------------------
#  Cache and set Log4cplus_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Log4cplus
  FOUND_VAR Log4cplus_FOUND
  REQUIRED_VARS
    Log4cplus_LIBRARY
    Log4cplus_INCLUDE_DIR
  VERSION_VAR Log4cplus_VERSION
)

if(NOT Log4cplus_FOUND)
  if(Log4cplus_FIND_REQUIRED)
    message(FATAL_ERROR "Unable to find Log4cplus")
  endif()
  return()
endif()

# Partition release/debug lib vars

set(Log4cplus_RELEASE_LIBRARIES ${Log4cplus_LIBRARY_RELEASE})
get_filename_component(Log4cplus_RELEASE_LIBRARY_DIRS ${Log4cplus_RELEASE_LIBRARIES} DIRECTORY)
set(Log4cplus_DEBUG_LIBRARIES ${Log4cplus_LIBRARY_DEBUG})
get_filename_component(Log4cplus_DEBUG_LIBRARY_DIRS ${Log4cplus_DEBUG_LIBRARIES} DIRECTORY)
set(Log4cplus_LIBRARIES ${Log4cplus_RELEASE_LIBRARIES})
set(Log4cplus_LIBRARY_DIRS ${Log4cplus_RELEASE_LIBRARY_DIRS})
set(Log4cplus_INCLUDE_DIRS ${Log4cplus_INCLUDE_DIR})

# Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
# lib is in use. If win32, we can't mark the import .libs as shared, so
# these are always marked as UNKNOWN. Otherwise, infer from extension.
set(LOG4CPLUS_LIB_TYPE UNKNOWN)
if(LOG4CPLUS_USE_STATIC_LIBS)
  set(LOG4CPLUS_LIB_TYPE STATIC)
elseif(UNIX)
  get_filename_component(_LOG4CPLUS_EXT ${Log4cplus_LIBRARY} EXT)
  if(_LOG4CPLUS_EXT STREQUAL ".a")
    set(LOG4CPLUS_LIB_TYPE STATIC)
  elseif(_LOG4CPLUS_EXT STREQUAL ".so" OR
         _LOG4CPLUS_EXT STREQUAL ".dylib")
    set(LOG4CPLUS_LIB_TYPE SHARED)
  endif()
endif()

get_filename_component(Log4cplus_LIBRARY_DIRS ${Log4cplus_LIBRARY_RELEASE} DIRECTORY)

if(NOT TARGET Log4cplus::log4cplus)
  add_library(Log4cplus::log4cplus ${LOG4CPLUS_LIB_TYPE} IMPORTED)
  set_target_properties(Log4cplus::log4cplus PROPERTIES
    INTERFACE_COMPILE_OPTIONS "${PC_Log4cplus_CFLAGS_OTHER}"
    INTERFACE_INCLUDE_DIRECTORIES "${Log4cplus_INCLUDE_DIRS}")

  # Standard location
  set_target_properties(Log4cplus::log4cplus PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX;RC"
    IMPORTED_LOCATION "${Log4cplus_LIBRARY}")

  # WIN32 APIs
  if(WIN32)
    set_target_properties(Log4cplus::log4cplus PROPERTIES
      IMPORTED_LINK_INTERFACE_LIBRARIES "ws2_32;advapi32")
  endif()

  # Release location
  if(EXISTS "${Log4cplus_LIBRARY_RELEASE}")
    set_property(TARGET Log4cplus::log4cplus APPEND PROPERTY
      IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(Log4cplus::log4cplus PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX;RC"
      IMPORTED_LOCATION_RELEASE "${Log4cplus_LIBRARY_RELEASE}")
  endif()

  # Debug location
  if(EXISTS "${Log4cplus_LIBRARY_DEBUG}")
    set_property(TARGET Log4cplus::log4cplus APPEND PROPERTY
      IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(Log4cplus::log4cplus PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX;RC"
      IMPORTED_LOCATION_DEBUG "${Log4cplus_LIBRARY_DEBUG}")
  endif()
endif()
