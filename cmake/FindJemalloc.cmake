# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[.rst:

FindJemalloc
-----------

Find Jemalloc include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(Jemalloc
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if Jemalloc is not found
    )


IMPORTED Targets
^^^^^^^^^^^^^^^^

``Jemalloc::jemalloc``
  This module defines IMPORTED target Jemalloc::jemalloc, if Jemalloc has been
  found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Jemalloc_FOUND``
  True if the system has the Jemalloc library.
``Jemalloc_VERSION``
  The version of the Jemalloc library which was found.
``Jemalloc_LIBRARIES``
  Libraries needed to link to Jemalloc.
``Jemalloc_LIBRARY_DIRS``
  Jemalloc library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Jemalloc_LIBRARY``
  The path to the Jemalloc library.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``Jemalloc_ROOT``
  Preferred installation prefix.
``JEMALLOC_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``JEMALLOC_USE_STATIC_LIBS``
  Only search for static jemalloc libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
include(GNUInstallDirs)


mark_as_advanced(
  Jemalloc_LIBRARY
)

set(_FIND_JEMALLOC_ADDITIONAL_OPTIONS "")
if(DISABLE_CMAKE_SEARCH_PATHS)
  set(_FIND_JEMALLOC_ADDITIONAL_OPTIONS NO_DEFAULT_PATH)
endif()

# Set _JEMALLOC_ROOT based on a user provided root var. Xxx_ROOT and ENV{Xxx_ROOT}
# are prioritised over the legacy capitalized XXX_ROOT variables for matching
# CMake 3.12 behaviour
# @todo  deprecate -D and ENV JEMALLOC_ROOT from CMake 3.12
if(Jemalloc_ROOT)
  set(_JEMALLOC_ROOT ${Jemalloc_ROOT})
elseif(DEFINED ENV{Jemalloc_ROOT})
  set(_JEMALLOC_ROOT $ENV{Jemalloc_ROOT})
elseif(JEMALLOC_ROOT)
  set(_JEMALLOC_ROOT ${JEMALLOC_ROOT})
elseif(DEFINED ENV{JEMALLOC_ROOT})
  set(_JEMALLOC_ROOT $ENV{JEMALLOC_ROOT})
endif()

# Additionally try and use pkconfig to find jemalloc

if(NOT DEFINED PKG_CONFIG_FOUND)
  find_package(PkgConfig)
endif()
pkg_check_modules(PC_Jemalloc QUIET jemalloc)

# ------------------------------------------------------------------------
#  Search for Jemalloc lib DIR
# ------------------------------------------------------------------------

set(_JEMALLOC_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _JEMALLOC_LIBRARYDIR_SEARCH_DIRS
  ${JEMALLOC_LIBRARYDIR}
  ${_JEMALLOC_ROOT}
  ${PC_Jemalloc_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Library suffix handling

set(_JEMALLOC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(WIN32)
  if(JEMALLOC_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  else()
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "_dll.lib")
  endif()
else()
  if(JEMALLOC_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
endif()

# Build suffix directories

set(JEMALLOC_PATH_SUFFIXES
  ${CMAKE_INSTALL_LIBDIR}
  lib64
  lib
)

find_library(Jemalloc_LIBRARY jemalloc
  ${_FIND_JEMALLOC_ADDITIONAL_OPTIONS}
  PATHS ${_JEMALLOC_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${JEMALLOC_PATH_SUFFIXES}
)

# Reset library suffix

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_JEMALLOC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_JEMALLOC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

# ------------------------------------------------------------------------
#  Cache and set Jemalloc_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Jemalloc
  FOUND_VAR Jemalloc_FOUND
  REQUIRED_VARS
    Jemalloc_LIBRARY
  VERSION_VAR Jemalloc_VERSION
)

if(Jemalloc_FOUND)
  # Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
  # lib is in use. If win32, we can't mark the import .libs as shared, so
  # these are always marked as UNKNOWN. Otherwise, infer from extension.
  set(JEMALLOC_LIB_TYPE UNKNOWN)
  if(JEMALLOC_USE_STATIC_LIBS)
    set(JEMALLOC_LIB_TYPE STATIC)
  elseif(UNIX)
    get_filename_component(_JEMALLOC_EXT ${Jemalloc_LIBRARY} EXT)
    if(_JEMALLOC_EXT STREQUAL ".a")
      set(JEMALLOC_LIB_TYPE STATIC)
    elseif(_JEMALLOC_EXT STREQUAL ".so" OR
           _JEMALLOC_EXT STREQUAL ".dylib")
      set(JEMALLOC_LIB_TYPE SHARED)
    endif()
  endif()

  set(Jemalloc_LIBRARIES ${Jemalloc_LIBRARY})

  get_filename_component(Jemalloc_LIBRARY_DIRS ${Jemalloc_LIBRARY} DIRECTORY)

  if(NOT TARGET Jemalloc::jemalloc)
    add_library(Jemalloc::jemalloc ${JEMALLOC_LIB_TYPE} IMPORTED)
    set_target_properties(Jemalloc::jemalloc PROPERTIES
      IMPORTED_LOCATION "${Jemalloc_LIBRARIES}"
      INTERFACE_COMPILE_OPTIONS "${PC_Jemalloc_CFLAGS_OTHER}"
    )
  endif()
elseif(Jemalloc_FIND_REQUIRED)
  message(FATAL_ERROR "Unable to find Jemalloc")
endif()
