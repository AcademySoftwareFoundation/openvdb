# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
#[=======================================================================[.rst:

FindZstd
--------

Find Zstd include dirs and libraries

The zstd CMake config files are used if they are available. Otherwise zstd is
located manually, which supports installs that do not ship the config files.

Use this module by invoking find_package with the form::

  find_package(Zstd
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if Zstd is not found
    )


IMPORTED Targets
^^^^^^^^^^^^^^^^

``Zstd::zstd``
  This module defines target Zstd::zstd, if Zstd has been found. When the zstd
  CMake config is used this is an alias of the config's static or shared
  target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Zstd_FOUND``
  True if the system has the Zstd library.
``Zstd_VERSION``
  The version of the Zstd library which was found.
``Zstd_INCLUDE_DIRS``
  Include directories needed to use Zstd.

Cache Variables
^^^^^^^^^^^^^^^

These are only set when the zstd CMake config is not used.

The following cache variables may also be set:

``Zstd_INCLUDE_DIR``
  The directory containing ``zstd.h``.
``Zstd_LIBRARY``
  The path to the Zstd library.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``Zstd_ROOT``
  Preferred installation prefix.
``ZSTD_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``ZSTD_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``ZSTD_USE_STATIC_LIBS``
  Prefer static zstd libraries. A warning is issued if only a shared library
  is available.
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.24)
include(GNUInstallDirs)
include(FindPackageHandleStandardArgs)

mark_as_advanced(
  Zstd_INCLUDE_DIR
  Zstd_LIBRARY
)

set(_FIND_ZSTD_ADDITIONAL_OPTIONS "")
if(DISABLE_CMAKE_SEARCH_PATHS)
  set(_FIND_ZSTD_ADDITIONAL_OPTIONS NO_DEFAULT_PATH)
endif()

# Set _ZSTD_ROOT based on a user provided root var. Xxx_ROOT and ENV{Xxx_ROOT}
# are prioritised over the capitalized XXX_ROOT variables
if(Zstd_ROOT)
  set(_ZSTD_ROOT ${Zstd_ROOT})
elseif(DEFINED ENV{Zstd_ROOT})
  set(_ZSTD_ROOT $ENV{Zstd_ROOT})
elseif(ZSTD_ROOT)
  set(_ZSTD_ROOT ${ZSTD_ROOT})
elseif(DEFINED ENV{ZSTD_ROOT})
  set(_ZSTD_ROOT $ENV{ZSTD_ROOT})
endif()

# A nested find_package call can overwrite variables of the caller, so run it
# in a function scope.
function(_find_zstd_config result_found result_version result_target)
  set(version "")
  if(Zstd_FIND_VERSION)
    set(version ${Zstd_FIND_VERSION})
    if(Zstd_FIND_VERSION_EXACT)
      list(APPEND version EXACT)
    endif()
  endif()
  find_package(zstd ${version} CONFIG QUIET
    HINTS ${_ZSTD_ROOT}
    ${_FIND_ZSTD_ADDITIONAL_OPTIONS}
  )
  set(target "")
  if(zstd_FOUND)
    if(TARGET zstd::libzstd_static AND (ZSTD_USE_STATIC_LIBS OR NOT TARGET zstd::libzstd_shared))
      set(target zstd::libzstd_static)
    elseif(TARGET zstd::libzstd_shared)
      set(target zstd::libzstd_shared)
      if(ZSTD_USE_STATIC_LIBS)
        message(WARNING "ZSTD_USE_STATIC_LIBS is set but only a shared zstd library was found.")
      endif()
    elseif(TARGET zstd::libzstd)
      set(target zstd::libzstd)
    endif()
  endif()
  set(${result_found} ${zstd_FOUND} PARENT_SCOPE)
  set(${result_version} ${zstd_VERSION} PARENT_SCOPE)
  set(${result_target} ${target} PARENT_SCOPE)
endfunction()

# Prefer the config files, unless the user pointed at a specific library
if(NOT Zstd_LIBRARY AND NOT Zstd_INCLUDE_DIR)
  _find_zstd_config(_ZSTD_CONFIG_FOUND _ZSTD_CONFIG_VERSION _ZSTD_CONFIG_TARGET)
  if(_ZSTD_CONFIG_FOUND AND _ZSTD_CONFIG_TARGET)
    set(Zstd_VERSION ${_ZSTD_CONFIG_VERSION})
    if(NOT TARGET Zstd::zstd)
      add_library(Zstd::zstd ALIAS ${_ZSTD_CONFIG_TARGET})
    endif()
    find_package_handle_standard_args(Zstd
      FOUND_VAR Zstd_FOUND
      REQUIRED_VARS Zstd_VERSION
      VERSION_VAR Zstd_VERSION
    )
    unset(_ZSTD_CONFIG_FOUND)
    unset(_ZSTD_CONFIG_VERSION)
    unset(_ZSTD_CONFIG_TARGET)
    return()
  endif()
  unset(_ZSTD_CONFIG_FOUND)
  unset(_ZSTD_CONFIG_VERSION)
  unset(_ZSTD_CONFIG_TARGET)
endif()

# Additionally try and use pkconfig to find zstd
if(USE_PKGCONFIG)
  if(NOT DEFINED PKG_CONFIG_FOUND)
    find_package(PkgConfig)
  endif()
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_Zstd QUIET libzstd)
  endif()
endif()

# ------------------------------------------------------------------------
#  Search for Zstd include DIR
# ------------------------------------------------------------------------

set(_ZSTD_INCLUDE_SEARCH_DIRS "")
list(APPEND _ZSTD_INCLUDE_SEARCH_DIRS
  ${ZSTD_INCLUDEDIR}
  ${_ZSTD_ROOT}
  ${PC_Zstd_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

find_path(Zstd_INCLUDE_DIR zstd.h
  ${_FIND_ZSTD_ADDITIONAL_OPTIONS}
  PATHS ${_ZSTD_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR} include
)
unset(_ZSTD_INCLUDE_SEARCH_DIRS)

if(EXISTS "${Zstd_INCLUDE_DIR}/zstd.h")
  foreach(_ZSTD_VERSION_PART MAJOR MINOR RELEASE)
    file(STRINGS "${Zstd_INCLUDE_DIR}/zstd.h"
      _zstd_version_string REGEX "#define ZSTD_VERSION_${_ZSTD_VERSION_PART} +[0-9]+"
    )
    string(REGEX REPLACE "#define ZSTD_VERSION_${_ZSTD_VERSION_PART} +([0-9]+).*$" "\\1"
      Zstd_VERSION_${_ZSTD_VERSION_PART} "${_zstd_version_string}"
    )
  endforeach()
  unset(_zstd_version_string)
  unset(_ZSTD_VERSION_PART)

  set(Zstd_VERSION ${Zstd_VERSION_MAJOR}.${Zstd_VERSION_MINOR}.${Zstd_VERSION_RELEASE})
endif()

# ------------------------------------------------------------------------
#  Search for Zstd library
# ------------------------------------------------------------------------

set(_ZSTD_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _ZSTD_LIBRARYDIR_SEARCH_DIRS
  ${ZSTD_LIBRARYDIR}
  ${_ZSTD_ROOT}
  ${PC_Zstd_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

set(_ZSTD_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(_ZSTD_LIB_NAMES zstd)
if(ZSTD_USE_STATIC_LIBS)
  if(MSVC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
    # The zstd CMake build names the static library zstd_static.lib on
    # Windows, the DLL import library is zstd.lib
    set(_ZSTD_LIB_NAMES zstd_static zstd)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
endif()

find_library(Zstd_LIBRARY
  NAMES ${_ZSTD_LIB_NAMES}
  NAMES_PER_DIR
  ${_FIND_ZSTD_ADDITIONAL_OPTIONS}
  PATHS ${_ZSTD_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
)

unset(_ZSTD_LIB_NAMES)
unset(_ZSTD_LIBRARYDIR_SEARCH_DIRS)
set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ZSTD_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_ZSTD_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

find_package_handle_standard_args(Zstd
  FOUND_VAR Zstd_FOUND
  REQUIRED_VARS
    Zstd_LIBRARY
    Zstd_INCLUDE_DIR
  VERSION_VAR Zstd_VERSION
)

if(NOT Zstd_FOUND)
  return()
endif()

set(Zstd_INCLUDE_DIRS ${Zstd_INCLUDE_DIR})

# ------------------------------------------------------------------------
#  Configure imported target
# ------------------------------------------------------------------------

if(NOT TARGET Zstd::zstd)
  # If win32, we can't mark the import .libs as shared, so these are always
  # UNKNOWN. Otherwise, infer from the extension.
  set(_ZSTD_LIB_TYPE UNKNOWN)
  if(ZSTD_USE_STATIC_LIBS)
    set(_ZSTD_LIB_TYPE STATIC)
  elseif(UNIX)
    get_filename_component(_ZSTD_EXT ${Zstd_LIBRARY} EXT)
    if(_ZSTD_EXT STREQUAL ".a")
      set(_ZSTD_LIB_TYPE STATIC)
    elseif(_ZSTD_EXT MATCHES "^\\.(so|dylib)")
      set(_ZSTD_LIB_TYPE SHARED)
    endif()
  endif()

  add_library(Zstd::zstd ${_ZSTD_LIB_TYPE} IMPORTED)
  set_target_properties(Zstd::zstd PROPERTIES
    INTERFACE_COMPILE_OPTIONS "${PC_Zstd_CFLAGS_OTHER}"
    INTERFACE_INCLUDE_DIRECTORIES "${Zstd_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${Zstd_LIBRARY}"
  )

  # A static zstd built with multithreading needs pthreads at link time
  if(_ZSTD_LIB_TYPE STREQUAL STATIC AND UNIX)
    find_package(Threads REQUIRED)
    set_property(TARGET Zstd::zstd APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES Threads::Threads)
  endif()

  unset(_ZSTD_LIB_TYPE)
  unset(_ZSTD_EXT)
endif()
