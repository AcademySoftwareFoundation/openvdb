# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
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

``Blosc_ROOT``
  Preferred installation prefix.
``BLOSC_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``BLOSC_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``BLOSC_USE_STATIC_LIBS``
  Only search for static blosc libraries
``BLOSC_USE_EXTERNAL_SOURCES``
  Set to ON if Blosc has been built using external sources for LZ4, snappy,
  zlib and zstd. Default is OFF.
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
include(GNUInstallDirs)

# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

mark_as_advanced(
  Blosc_INCLUDE_DIR
  Blosc_LIBRARY
)

set(_FIND_BLOSC_ADDITIONAL_OPTIONS "")
if(DISABLE_CMAKE_SEARCH_PATHS)
  set(_FIND_BLOSC_ADDITIONAL_OPTIONS NO_DEFAULT_PATH)
endif()

# Set _BLOSC_ROOT based on a user provided root var. Xxx_ROOT and ENV{Xxx_ROOT}
# are prioritised over the legacy capitalized XXX_ROOT variables for matching
# CMake 3.12 behaviour
# @todo  deprecate -D and ENV BLOSC_ROOT from CMake 3.12
if(Blosc_ROOT)
  set(_BLOSC_ROOT ${Blosc_ROOT})
elseif(DEFINED ENV{Blosc_ROOT})
  set(_BLOSC_ROOT $ENV{Blosc_ROOT})
elseif(BLOSC_ROOT)
  set(_BLOSC_ROOT ${BLOSC_ROOT})
elseif(DEFINED ENV{BLOSC_ROOT})
  set(_BLOSC_ROOT $ENV{BLOSC_ROOT})
endif()

# Additionally try and use pkconfig to find blosc
if(USE_PKGCONFIG)
  if(NOT DEFINED PKG_CONFIG_FOUND)
    find_package(PkgConfig)
  endif()
  pkg_check_modules(PC_Blosc QUIET blosc)
endif()

# ------------------------------------------------------------------------
#  Search for blosc include DIR
# ------------------------------------------------------------------------

set(_BLOSC_INCLUDE_SEARCH_DIRS "")
list(APPEND _BLOSC_INCLUDE_SEARCH_DIRS
  ${BLOSC_INCLUDEDIR}
  ${_BLOSC_ROOT}
  ${PC_Blosc_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard blosc header file.
find_path(Blosc_INCLUDE_DIR blosc.h
  ${_FIND_BLOSC_ADDITIONAL_OPTIONS}
  PATHS ${_BLOSC_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR} include
)

if(EXISTS "${Blosc_INCLUDE_DIR}/blosc.h")
  file(STRINGS "${Blosc_INCLUDE_DIR}/blosc.h"
    _blosc_version_major_string REGEX "#define BLOSC_VERSION_MAJOR +[0-9]+ "
  )
  string(REGEX REPLACE "#define BLOSC_VERSION_MAJOR +([0-9]+).*$" "\\1"
    _blosc_version_major_string "${_blosc_version_major_string}"
  )
  string(STRIP "${_blosc_version_major_string}" Blosc_VERSION_MAJOR)

  file(STRINGS "${Blosc_INCLUDE_DIR}/blosc.h"
     _blosc_version_minor_string REGEX "#define BLOSC_VERSION_MINOR +[0-9]+ "
  )
  string(REGEX REPLACE "#define BLOSC_VERSION_MINOR +([0-9]+).*$" "\\1"
    _blosc_version_minor_string "${_blosc_version_minor_string}"
  )
  string(STRIP "${_blosc_version_minor_string}" Blosc_VERSION_MINOR)

  file(STRINGS "${Blosc_INCLUDE_DIR}/blosc.h"
     _blosc_version_release_string REGEX "#define BLOSC_VERSION_RELEASE +[0-9]+ "
  )
  string(REGEX REPLACE "#define BLOSC_VERSION_RELEASE +([0-9]+).*$" "\\1"
    _blosc_version_release_string "${_blosc_version_release_string}"
  )
  string(STRIP "${_blosc_version_release_string}" Blosc_VERSION_RELEASE)

  unset(_blosc_version_major_string)
  unset(_blosc_version_minor_string)
  unset(_blosc_version_release_string)

  set(Blosc_VERSION ${Blosc_VERSION_MAJOR}.${Blosc_VERSION_MINOR}.${Blosc_VERSION_RELEASE})
endif()

# ------------------------------------------------------------------------
#  Search for blosc lib DIR
# ------------------------------------------------------------------------

set(_BLOSC_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _BLOSC_LIBRARYDIR_SEARCH_DIRS
  ${BLOSC_LIBRARYDIR}
  ${_BLOSC_ROOT}
  ${PC_Blosc_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
)

# Library suffix handling

set(_BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(WIN32)
  if(BLOSC_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  endif()
else()
  if(BLOSC_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
endif()

# libblosc is the name of the blosc static lib on windows

find_library(Blosc_LIBRARY blosc libblosc
  ${_FIND_BLOSC_ADDITIONAL_OPTIONS}
  PATHS ${_BLOSC_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
)

# Reset library suffix

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

# ------------------------------------------------------------------------
#  Cache and set Blosc_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Blosc
  FOUND_VAR Blosc_FOUND
  REQUIRED_VARS
    Blosc_LIBRARY
    Blosc_INCLUDE_DIR
  VERSION_VAR Blosc_VERSION
)

if(Blosc_FOUND)
  # Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
  # lib is in use. If win32, we can't mark the import .libs as shared, so
  # these are always marked as UNKNOWN. Otherwise, infer from extension.
  set(BLOSC_LIB_TYPE UNKNOWN)
  if(BLOSC_USE_STATIC_LIBS)
    set(BLOSC_LIB_TYPE STATIC)
  elseif(UNIX)
    get_filename_component(_BLOSC_EXT ${Blosc_LIBRARY} EXT)
    if(_BLOSC_EXT STREQUAL ".a")
      set(BLOSC_LIB_TYPE STATIC)
    elseif(_BLOSC_EXT STREQUAL ".so" OR
           _BLOSC_EXT STREQUAL ".dylib")
      set(BLOSC_LIB_TYPE SHARED)
    endif()
  endif()

  set(Blosc_LIBRARIES ${Blosc_LIBRARY})
  set(Blosc_INCLUDE_DIRS ${Blosc_INCLUDE_DIR})

  get_filename_component(Blosc_LIBRARY_DIRS ${Blosc_LIBRARY} DIRECTORY)

  if(NOT TARGET Blosc::blosc)
    add_library(Blosc::blosc ${BLOSC_LIB_TYPE} IMPORTED)
    set_target_properties(Blosc::blosc PROPERTIES
      IMPORTED_LOCATION "${Blosc_LIBRARIES}"
      INTERFACE_COMPILE_OPTIONS "${PC_Blosc_CFLAGS_OTHER}"
      INTERFACE_INCLUDE_DIRECTORIES "${Blosc_INCLUDE_DIRS}"
    )

    # Blosc may optionally be compiled with external sources for
    # lz4, snappy, zlib and zstd. Add them as interface libs if
    # requested (there doesn't seem to be a way to figure this
    # out automatically).
    if(BLOSC_USE_EXTERNAL_SOURCES)
      set_target_properties(Blosc::blosc PROPERTIES
        INTERFACE_LINK_DIRECTORIES "${Blosc_LIBRARY_DIRS}"
        INTERFACE_LINK_LIBRARIES "lz4;snappy;zlib;zstd_static"
      )
    endif()
  endif()
elseif(Blosc_FIND_REQUIRED)
  message(FATAL_ERROR "Unable to find Blosc")
endif()
