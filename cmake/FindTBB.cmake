# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
#[=======================================================================[.rst:

FindTBB
-------

Find oneTBB include dirs and libraries. Only oneTBB (TBB 2021 and newer) is
supported.

The search order is:

1. The CMake package config files shipped with oneTBB (``TBBConfig.cmake``).
   This is skipped if any of the explicit location hints listed below are set.
2. A manual search for headers and libraries, using pkg-config (when
   ``USE_PKGCONFIG`` is enabled) and the hints listed below.

Use this module by invoking find_package with the form::

  find_package(TBB
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if Tbb is not found
    [COMPONENTS <libs>...] # Tbb libraries by their canonical name
                           # e.g. "tbb" for "libtbb"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``TBB::tbb``
  The tbb library target.
``TBB::tbbmalloc``
  The tbbmalloc library target.
``TBB::tbbmalloc_proxy``
  The tbbmalloc_proxy library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``TBB_FOUND``
  True if the system has the Tbb library.
``TBB_VERSION`` ``Tbb_VERSION``
  The version of the Tbb library which was found.
``Tbb_INCLUDE_DIRS``
  Include directories needed to use Tbb. Only set when the manual search is
  used, prefer the imported targets.
``TBB_{COMPONENT}_FOUND``
  True if the system has the named TBB component.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set by the manual search:

``Tbb_INCLUDE_DIR``
  The directory containing ``oneapi/tbb/version.h``.
``Tbb_{COMPONENT}_LIBRARY_RELEASE``
  Individual component libraries for Tbb release
``Tbb_{COMPONENT}_LIBRARY_DEBUG``
  Individual debug component libraries for Tbb debug

Hints
^^^^^

``TBB_ROOT``
  Preferred installation prefix. Used by both the config and manual search.
``TBB_DIR``
  Directory containing ``TBBConfig.cmake``.
``TBB_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include. Setting this forces the
  manual search.
``TBB_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib. Setting this forces the
  manual search.
``TBB_DEBUG_SUFFIX``
  Suffix of the debug version of tbb. Defaults to "_debug".
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``TBB_USE_STATIC_LIBS``
  Only search for static tbb libraries in the manual search.
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.24)
include(GNUInstallDirs)
include(FindPackageHandleStandardArgs)

mark_as_advanced(Tbb_INCLUDE_DIR)

set(_FIND_TBB_ADDITIONAL_OPTIONS "")
if(DISABLE_CMAKE_SEARCH_PATHS)
  set(_FIND_TBB_ADDITIONAL_OPTIONS NO_DEFAULT_PATH)
endif()

set(_TBB_COMPONENT_LIST
  tbb
  tbbmalloc
  tbbmalloc_proxy
)

if(TBB_FIND_COMPONENTS)
  set(_IGNORED_COMPONENTS "")
  foreach(COMPONENT ${TBB_FIND_COMPONENTS})
    if(NOT ${COMPONENT} IN_LIST _TBB_COMPONENT_LIST)
      list(APPEND _IGNORED_COMPONENTS ${COMPONENT})
    endif()
  endforeach()

  if(_IGNORED_COMPONENTS)
    message(STATUS "Ignoring unknown components of TBB:")
    foreach(COMPONENT ${_IGNORED_COMPONENTS})
      message(STATUS "  ${COMPONENT}")
    endforeach()
    list(REMOVE_ITEM TBB_FIND_COMPONENTS ${_IGNORED_COMPONENTS})
  endif()
  unset(_IGNORED_COMPONENTS)
else()
  set(TBB_FIND_COMPONENTS ${_TBB_COMPONENT_LIST})
endif()

if(TBB_ROOT)
  set(_TBB_ROOT ${TBB_ROOT})
elseif(DEFINED ENV{TBB_ROOT})
  set(_TBB_ROOT $ENV{TBB_ROOT})
endif()

# ------------------------------------------------------------------------
#  1. Search for the oneTBB CMake config package
# ------------------------------------------------------------------------

# Explicit locations indicate the user wants a specific installation that may
# not ship a config file (e.g. a DCC application), so go straight to the
# manual search. Static builds also use the manual search so that only static
# libraries are considered.
set(_TBB_USE_CONFIG ON)
if(TBB_INCLUDEDIR OR TBB_LIBRARYDIR OR Tbb_INCLUDE_DIR OR TBB_USE_STATIC_LIBS)
  set(_TBB_USE_CONFIG OFF)
endif()
foreach(COMPONENT ${TBB_FIND_COMPONENTS})
  if(Tbb_${COMPONENT}_LIBRARY_RELEASE OR Tbb_${COMPONENT}_LIBRARY_DEBUG)
    set(_TBB_USE_CONFIG OFF)
  endif()
endforeach()

# A nested find_package call overwrites the TBB_FIND_* variables of the
# caller, so run it in a function scope.
function(_find_tbb_config result_found result_version)
  set(version "")
  if(TBB_FIND_VERSION)
    set(version ${TBB_FIND_VERSION})
    if(TBB_FIND_VERSION_EXACT)
      list(APPEND version EXACT)
    endif()
  endif()
  find_package(TBB ${version} CONFIG QUIET
    COMPONENTS ${TBB_FIND_COMPONENTS}
    HINTS ${_TBB_ROOT}
    ${_FIND_TBB_ADDITIONAL_OPTIONS}
  )
  set(${result_found} ${TBB_FOUND} PARENT_SCOPE)
  set(${result_version} ${TBB_VERSION} PARENT_SCOPE)
endfunction()

if(_TBB_USE_CONFIG)
  _find_tbb_config(_TBB_CONFIG_FOUND _TBB_CONFIG_VERSION)
  if(_TBB_CONFIG_FOUND)
    set(TBB_VERSION ${_TBB_CONFIG_VERSION})
    set(Tbb_VERSION ${TBB_VERSION})
    foreach(COMPONENT ${TBB_FIND_COMPONENTS})
      if(TARGET TBB::${COMPONENT})
        set(TBB_${COMPONENT}_FOUND TRUE)
      else()
        set(TBB_${COMPONENT}_FOUND FALSE)
      endif()
    endforeach()

    find_package_handle_standard_args(TBB
      REQUIRED_VARS TBB_VERSION
      VERSION_VAR TBB_VERSION
      HANDLE_COMPONENTS
    )
    unset(_TBB_USE_CONFIG)
    unset(_TBB_CONFIG_FOUND)
    unset(_TBB_CONFIG_VERSION)
    return()
  endif()
  unset(_TBB_CONFIG_FOUND)
  unset(_TBB_CONFIG_VERSION)
endif()
unset(_TBB_USE_CONFIG)

# ------------------------------------------------------------------------
#  2. Manual search, with optional pkg-config hints
# ------------------------------------------------------------------------

if(USE_PKGCONFIG)
  if(NOT DEFINED PKG_CONFIG_FOUND)
    find_package(PkgConfig)
  endif()
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_Tbb QUIET tbb)
  endif()
endif()

# Include directory

set(_TBB_INCLUDE_SEARCH_DIRS "")
list(APPEND _TBB_INCLUDE_SEARCH_DIRS
  ${TBB_INCLUDEDIR}
  ${_TBB_ROOT}
  ${PC_Tbb_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
)

# include/tbb is an alias of include/oneapi/tbb, so locate the real header
find_path(Tbb_INCLUDE_DIR oneapi/tbb/version.h
  ${_FIND_TBB_ADDITIONAL_OPTIONS}
  PATHS ${_TBB_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR} include
)
unset(_TBB_INCLUDE_SEARCH_DIRS)

set(_tbb_version_file "${Tbb_INCLUDE_DIR}/oneapi/tbb/version.h")
if(EXISTS ${_tbb_version_file})
  foreach(_tbb_define TBB_VERSION_MAJOR TBB_VERSION_MINOR __TBB_BINARY_VERSION)
    file(STRINGS ${_tbb_version_file} _tbb_line REGEX "#define ${_tbb_define} ")
    string(REGEX REPLACE ".*#define ${_tbb_define} +([0-9]+).*" "\\1"
      _tbb_${_tbb_define} "${_tbb_line}")
  endforeach()

  set(Tbb_BINARY_VERSION ${_tbb___TBB_BINARY_VERSION})
  # Set both for compatibility reasons, TBB's CONFIG files only set the latter
  set(Tbb_VERSION ${_tbb_TBB_VERSION_MAJOR}.${_tbb_TBB_VERSION_MINOR})
  set(TBB_VERSION ${Tbb_VERSION})
  unset(_tbb_line)
  unset(_tbb_define)
endif()
unset(_tbb_version_file)

# Libraries

set(_TBB_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _TBB_LIBRARYDIR_SEARCH_DIRS
  ${TBB_LIBRARYDIR}
  ${_TBB_ROOT}
  ${PC_Tbb_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
)

if(NOT DEFINED TBB_DEBUG_SUFFIX)
  set(TBB_DEBUG_SUFFIX _debug)
endif()

set(_TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
if(TBB_USE_STATIC_LIBS)
  if(MSVC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
endif()

set(Tbb_LIB_COMPONENTS "")
foreach(COMPONENT ${TBB_FIND_COMPONENTS})
  foreach(BUILD_TYPE RELEASE DEBUG)
    set(_TBB_SUFFIX "")
    if(BUILD_TYPE STREQUAL DEBUG)
      set(_TBB_SUFFIX ${TBB_DEBUG_SUFFIX})
    endif()

    # Some platforms version the library name, e.g. tbb12.lib. Try the
    # unversioned name first.
    set(_TBB_LIB_NAMES ${COMPONENT}${_TBB_SUFFIX})
    if(Tbb_BINARY_VERSION)
      list(APPEND _TBB_LIB_NAMES ${COMPONENT}${Tbb_BINARY_VERSION}${_TBB_SUFFIX})
    endif()

    find_library(Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE}
      NAMES ${_TBB_LIB_NAMES}
      ${_FIND_TBB_ADDITIONAL_OPTIONS}
      PATHS ${_TBB_LIBRARYDIR_SEARCH_DIRS}
      PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
    )
    mark_as_advanced(Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE})
  endforeach()

  # If only one build type was found, use it for both
  if(Tbb_${COMPONENT}_LIBRARY_RELEASE AND NOT Tbb_${COMPONENT}_LIBRARY_DEBUG)
    set(Tbb_${COMPONENT}_LIBRARY_DEBUG ${Tbb_${COMPONENT}_LIBRARY_RELEASE})
  elseif(Tbb_${COMPONENT}_LIBRARY_DEBUG AND NOT Tbb_${COMPONENT}_LIBRARY_RELEASE)
    set(Tbb_${COMPONENT}_LIBRARY_RELEASE ${Tbb_${COMPONENT}_LIBRARY_DEBUG})
  endif()

  if(Tbb_${COMPONENT}_LIBRARY_RELEASE)
    set(TBB_${COMPONENT}_FOUND TRUE)
    list(APPEND Tbb_LIB_COMPONENTS ${Tbb_${COMPONENT}_LIBRARY_RELEASE})
  else()
    set(TBB_${COMPONENT}_FOUND FALSE)
  endif()
endforeach()
unset(_TBB_SUFFIX)
unset(_TBB_LIB_NAMES)
unset(_TBB_LIBRARYDIR_SEARCH_DIRS)

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

find_package_handle_standard_args(TBB
  FOUND_VAR TBB_FOUND
  REQUIRED_VARS
    Tbb_INCLUDE_DIR
    Tbb_LIB_COMPONENTS
  VERSION_VAR Tbb_VERSION
  HANDLE_COMPONENTS
)

if(NOT TBB_FOUND)
  return()
endif()

set(Tbb_INCLUDE_DIRS ${Tbb_INCLUDE_DIR})

# ------------------------------------------------------------------------
#  Configure imported targets
# ------------------------------------------------------------------------

foreach(COMPONENT ${TBB_FIND_COMPONENTS})
  if(TARGET TBB::${COMPONENT})
    continue()
  endif()

  # If win32, we can't mark the import .libs as shared, so these are always
  # UNKNOWN. Otherwise, infer from the extension.
  set(_TBB_LIB_TYPE UNKNOWN)
  if(TBB_USE_STATIC_LIBS)
    set(_TBB_LIB_TYPE STATIC)
  elseif(UNIX)
    get_filename_component(_TBB_EXT ${Tbb_${COMPONENT}_LIBRARY_RELEASE} EXT)
    if(_TBB_EXT STREQUAL ".a")
      set(_TBB_LIB_TYPE STATIC)
    elseif(_TBB_EXT MATCHES "^\\.(so|dylib)")
      set(_TBB_LIB_TYPE SHARED)
    endif()
  endif()

  # Static libraries on Windows need to disable the auto-link pragma
  set(_TBB_DEFINITIONS "")
  if(WIN32 AND _TBB_LIB_TYPE STREQUAL STATIC)
    if(COMPONENT STREQUAL tbb)
      list(APPEND _TBB_DEFINITIONS __TBB_NO_IMPLICIT_LINKAGE=1)
    else()
      list(APPEND _TBB_DEFINITIONS __TBB_MALLOC_NO_IMPLICIT_LINKAGE=1)
    endif()
  endif()

  add_library(TBB::${COMPONENT} ${_TBB_LIB_TYPE} IMPORTED)
  set_target_properties(TBB::${COMPONENT} PROPERTIES
    INTERFACE_COMPILE_OPTIONS "${PC_Tbb_CFLAGS_OTHER}"
    INTERFACE_COMPILE_DEFINITIONS "${_TBB_DEFINITIONS}"
    INTERFACE_INCLUDE_DIRECTORIES "${Tbb_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_CONFIGURATIONS "RELEASE;DEBUG"
    IMPORTED_LOCATION_RELEASE "${Tbb_${COMPONENT}_LIBRARY_RELEASE}"
    IMPORTED_LOCATION_DEBUG "${Tbb_${COMPONENT}_LIBRARY_DEBUG}"
    MAP_IMPORTED_CONFIG_MINSIZEREL Release
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
  )
endforeach()
unset(_TBB_LIB_TYPE)
unset(_TBB_EXT)
unset(_TBB_DEFINITIONS)
