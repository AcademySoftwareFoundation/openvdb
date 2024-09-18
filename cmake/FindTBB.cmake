# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[.rst:

FindTBB
-------

Find Tbb include dirs and libraries

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

``Tbb_FOUND``
  True if the system has the Tbb library.
``Tbb_VERSION`` ``TBB_VERSION``
  The version of the Tbb library which was found.
``Tbb_INCLUDE_DIRS``
  Include directories needed to use Tbb.
``Tbb_RELEASE_LIBRARIES``
  Libraries needed to link to the release version of Tbb.
``Tbb_RELEASE_LIBRARY_DIRS``
  Tbb release library directories.
``Tbb_DEBUG_LIBRARIES``
  Libraries needed to link to the debug version of Tbb.
``Tbb_DEBUG_LIBRARY_DIRS``
  Tbb debug library directories.
``TBB_{COMPONENT}_FOUND``
  True if the system has the named TBB component.

Deprecated - use [RELEASE|DEBUG] variants:

``Tbb_LIBRARIES``
  Libraries needed to link to Tbb.
``Tbb_LIBRARY_DIRS``
  Tbb library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Tbb_INCLUDE_DIR``
  The directory containing ``tbb/tbb_stddef.h``.
``Tbb_{COMPONENT}_LIBRARY``
  Individual component libraries for Tbb. may include target_link_libraries() debug/optimized keywords.
``Tbb_{COMPONENT}_LIBRARY_RELEASE``
  Individual component libraries for Tbb release
``Tbb_{COMPONENT}_LIBRARY_DEBUG``
  Individual debug component libraries for Tbb debug

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``TBB_ROOT``
  Preferred installation prefix.
``TBB_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``TBB_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``TBB_DEBUG_SUFFIX``
  Suffix of the debug version of tbb. Defaults to "_debug".
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``TBB_USE_STATIC_LIBS``
  Only search for static tbb libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
include(GNUInstallDirs)


mark_as_advanced(
  Tbb_INCLUDE_DIR
  Tbb_LIBRARY
)

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
  set(_TBB_COMPONENTS_PROVIDED TRUE)
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
else()
  set(_TBB_COMPONENTS_PROVIDED FALSE)
  set(TBB_FIND_COMPONENTS ${_TBB_COMPONENT_LIST})
endif()

if(TBB_ROOT)
  set(_TBB_ROOT ${TBB_ROOT})
elseif(DEFINED ENV{TBB_ROOT})
  set(_TBB_ROOT $ENV{TBB_ROOT})
endif()

# Additionally try and use pkconfig to find Tbb
if(USE_PKGCONFIG)
  if(NOT DEFINED PKG_CONFIG_FOUND)
    find_package(PkgConfig)
  endif()
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_Tbb QUIET tbb)
  endif()
endif()

# ------------------------------------------------------------------------
#  Search for tbb include DIR
# ------------------------------------------------------------------------

set(_TBB_INCLUDE_SEARCH_DIRS "")
list(APPEND _TBB_INCLUDE_SEARCH_DIRS
  ${TBB_INCLUDEDIR}
  ${_TBB_ROOT}
  ${PC_Tbb_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
)

if(NOT Tbb_INCLUDE_DIR)
  # Look for a legacy tbb header file.
  find_path(Tbb_LEGACY_INCLUDE_DIR tbb/tbb_stddef.h
    ${_FIND_TBB_ADDITIONAL_OPTIONS}
    PATHS ${_TBB_INCLUDE_SEARCH_DIRS}
    PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR} include
  )
else()
  set(Tbb_LEGACY_INCLUDE_DIR ${Tbb_INCLUDE_DIR})
endif()

# Look for a new tbb header installation
# From TBB 2021, tbb_stddef is removed and the directory include/tbb is
# simply an alias for include/oneapi/tbb. Try and find the version header
# in oneapi/tbb
find_path(Tbb_INCLUDE_DIR oneapi/tbb/version.h
  ${_FIND_TBB_ADDITIONAL_OPTIONS}
  PATHS ${_TBB_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR} include
)

set(_tbb_legacy_version_file "${Tbb_LEGACY_INCLUDE_DIR}/tbb/tbb_stddef.h")
set(_tbb_version_file "${Tbb_INCLUDE_DIR}/oneapi/tbb/version.h")

if(EXISTS ${_tbb_legacy_version_file})
  if(EXISTS ${_tbb_version_file})
    message(WARNING "
      FindTBB located both an old and new tbb installation.
          old: ${_tbb_legacy_version_file}
          new: ${_tbb_version_file}
      The NEWER versioned installation will be used. You can set TBB_INCLUDEDIR
      to control FindTBB.cmake search, or explicitly set Tbb_INCLUDE_DIR to the
      desired location.
      ")
  else()
    set(_tbb_version_file "${_tbb_legacy_version_file}")
    set(Tbb_INCLUDE_DIR ${Tbb_LEGACY_INCLUDE_DIR} CACHE STRING "" FORCE)
  endif()
endif()


if(EXISTS ${_tbb_version_file})
  file(STRINGS ${_tbb_version_file} _tbb_version_major_string REGEX "#define TBB_VERSION_MAJOR " )
  string(REGEX REPLACE "#define TBB_VERSION_MAJOR" "" _tbb_version_major_string "${_tbb_version_major_string}")
  string(STRIP "${_tbb_version_major_string}" Tbb_VERSION_MAJOR)

  file(STRINGS ${_tbb_version_file} _tbb_version_minor_string REGEX "#define TBB_VERSION_MINOR ")
  string(REGEX REPLACE "#define TBB_VERSION_MINOR" "" _tbb_version_minor_string "${_tbb_version_minor_string}")
  string(STRIP "${_tbb_version_minor_string}" Tbb_VERSION_MINOR)

  file(STRINGS ${_tbb_version_file} _tbb_binary_version_string REGEX "#define __TBB_BINARY_VERSION ")
  string(REGEX REPLACE "#define __TBB_BINARY_VERSION" "" _tbb_binary_version_string "${_tbb_binary_version_string}")
  string(STRIP "${_tbb_binary_version_string}" Tbb_BINARY_VERSION)

  unset(_tbb_version_major_string)
  unset(_tbb_version_minor_string)
  unset(_tbb_binary_version_string)

  # Set both for compatibility reasons, TBB's CONFIG files only set the latter
  set(Tbb_VERSION ${Tbb_VERSION_MAJOR}.${Tbb_VERSION_MINOR})
  set(TBB_VERSION ${Tbb_VERSION})
endif()

unset(_tbb_version_file)
unset(_tbb_legacy_version_file)
unset(Tbb_LEGACY_INCLUDE_DIR)

# ------------------------------------------------------------------------
#  Search for TBB lib DIR
# ------------------------------------------------------------------------

set(_TBB_LIBRARYDIR_SEARCH_DIRS "")

# Append to _TBB_LIBRARYDIR_SEARCH_DIRS in priority order

set(_TBB_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _TBB_LIBRARYDIR_SEARCH_DIRS
  ${TBB_LIBRARYDIR}
  ${_TBB_ROOT}
  ${PC_Tbb_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
)

# Library suffix handling

if(NOT DEFINED TBB_DEBUG_SUFFIX)
  set(TBB_DEBUG_SUFFIX _debug)
endif()
set(_TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(MSVC)
  if(TBB_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  endif()
else()
  if(TBB_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
endif()

set(Tbb_LIB_COMPONENTS "")
list(APPEND TBB_BUILD_TYPES RELEASE DEBUG)

foreach(COMPONENT ${TBB_FIND_COMPONENTS})
  foreach(BUILD_TYPE ${TBB_BUILD_TYPES})

    set(_TBB_LIB_NAME ${COMPONENT})
    if(BUILD_TYPE STREQUAL DEBUG)
      set(_TBB_LIB_NAME "${_TBB_LIB_NAME}${TBB_DEBUG_SUFFIX}")
    endif()

    find_library(Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE} ${_TBB_LIB_NAME}
      ${_FIND_TBB_ADDITIONAL_OPTIONS}
      PATHS ${_TBB_LIBRARYDIR_SEARCH_DIRS}
      PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib)

    # If we didn't find the library, prepend Tbb_BINARY_VERSION to each possible
    # component name and try again. As of TBB 2021, TBB decides to version some
    # of its libraries on some of its platforms...
    if(NOT Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE} AND Tbb_BINARY_VERSION)
      set(_TBB_LIB_NAME "${COMPONENT}${Tbb_BINARY_VERSION}")
      if(BUILD_TYPE STREQUAL DEBUG)
        set(_TBB_LIB_NAME "${_TBB_LIB_NAME}${TBB_DEBUG_SUFFIX}")
      endif()

      find_library(Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE} ${_TBB_LIB_NAME}
        ${_FIND_TBB_ADDITIONAL_OPTIONS}
        PATHS ${_TBB_LIBRARYDIR_SEARCH_DIRS}
        PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib)
    endif()

    # On Unix, TBB sometimes uses linker scripts instead of symlinks, so parse the linker script
    # and correct the library name if so
    if(UNIX AND EXISTS ${Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
      # Ignore files where the first four bytes equals the ELF magic number
      file(READ ${Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE}} Tbb_${COMPONENT}_HEX OFFSET 0 LIMIT 4 HEX)
      if(NOT ${Tbb_${COMPONENT}_HEX} STREQUAL "7f454c46")
        # Read the first 1024 bytes of the library and match against an "INPUT (file)" regex
        file(READ ${Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE}} Tbb_${COMPONENT}_ASCII OFFSET 0 LIMIT 1024)
        if("${Tbb_${COMPONENT}_ASCII}" MATCHES "INPUT \\(([^(]+)\\)")
          # Extract the directory and apply the matched text (in brackets)
          get_filename_component(Tbb_${COMPONENT}_DIR "${Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE}}" DIRECTORY)
          set(Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE} "${Tbb_${COMPONENT}_DIR}/${CMAKE_MATCH_1}")
        endif()
      endif()
    endif()

    if(EXISTS ${Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
      list(APPEND Tbb_LIB_COMPONENTS ${Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
      list(APPEND Tbb_LIB_COMPONENTS_${BUILD_TYPE} ${Tbb_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
    endif()
  endforeach()

  if(Tbb_${COMPONENT}_LIBRARY_DEBUG AND Tbb_${COMPONENT}_LIBRARY_RELEASE)
    # if the generator is multi-config or if CMAKE_BUILD_TYPE is set for
    # single-config generators, set optimized and debug libraries
    get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_isMultiConfig OR CMAKE_BUILD_TYPE)
      set(Tbb_${COMPONENT}_LIBRARY optimized ${Tbb_${COMPONENT}_LIBRARY_RELEASE} debug ${Tbb_${COMPONENT}_LIBRARY_DEBUG})
    else()
      # For single-config generators where CMAKE_BUILD_TYPE has no value,
      # just use the release libraries
      set(Tbb_${COMPONENT}_LIBRARY ${Tbb_${COMPONENT}_LIBRARY_RELEASE})
    endif()
    # FIXME: This probably should be set for both cases
    set(Tbb_${COMPONENT}_LIBRARIES optimized ${Tbb_${COMPONENT}_LIBRARY_RELEASE} debug ${Tbb_${COMPONENT}_LIBRARY_DEBUG})
  endif()

  # if only the release version was found, set the debug variable also to the release version
  if(Tbb_${COMPONENT}_LIBRARY_RELEASE AND NOT Tbb_${COMPONENT}_LIBRARY_DEBUG)
    set(Tbb_${COMPONENT}_LIBRARY_DEBUG ${Tbb_${COMPONENT}_LIBRARY_RELEASE})
    set(Tbb_${COMPONENT}_LIBRARY       ${Tbb_${COMPONENT}_LIBRARY_RELEASE})
    set(Tbb_${COMPONENT}_LIBRARIES     ${Tbb_${COMPONENT}_LIBRARY_RELEASE})
  endif()

  # if only the debug version was found, set the release variable also to the debug version
  if(Tbb_${COMPONENT}_LIBRARY_DEBUG AND NOT Tbb_${COMPONENT}_LIBRARY_RELEASE)
    set(Tbb_${COMPONENT}_LIBRARY_RELEASE ${Tbb_${COMPONENT}_LIBRARY_DEBUG})
    set(Tbb_${COMPONENT}_LIBRARY         ${Tbb_${COMPONENT}_LIBRARY_DEBUG})
    set(Tbb_${COMPONENT}_LIBRARIES       ${Tbb_${COMPONENT}_LIBRARY_DEBUG})
  endif()

  # If the debug & release library ends up being the same, omit the keywords
  if("${Tbb_${COMPONENT}_LIBRARY_RELEASE}" STREQUAL "${Tbb_${COMPONENT}_LIBRARY_DEBUG}")
    set(Tbb_${COMPONENT}_LIBRARY   ${Tbb_${COMPONENT}_LIBRARY_RELEASE} )
    set(Tbb_${COMPONENT}_LIBRARIES ${Tbb_${COMPONENT}_LIBRARY_RELEASE} )
  endif()

  if(Tbb_${COMPONENT}_LIBRARY)
    set(TBB_${COMPONENT}_FOUND TRUE)
  else()
    set(TBB_${COMPONENT}_FOUND FALSE)
  endif()
endforeach()

# Reset library suffix

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

# ------------------------------------------------------------------------
#  Cache and set TBB_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB
  FOUND_VAR TBB_FOUND
  REQUIRED_VARS
    Tbb_INCLUDE_DIR
    Tbb_LIB_COMPONENTS
  VERSION_VAR Tbb_VERSION
  HANDLE_COMPONENTS
)

if(NOT TBB_FOUND)
  if(TBB_FIND_REQUIRED)
    message(FATAL_ERROR "Unable to find TBB")
  endif()
  return()
endif()

# Partition release/debug lib vars

set(Tbb_RELEASE_LIBRARIES "")
set(Tbb_RELEASE_LIBRARY_DIRS "")
set(Tbb_DEBUG_LIBRARIES "")
set(Tbb_DEBUG_LIBRARY_DIRS "")
foreach(LIB ${Tbb_LIB_COMPONENTS_RELEASE})
  get_filename_component(_TBB_LIBDIR ${LIB} DIRECTORY)
  list(APPEND Tbb_RELEASE_LIBRARIES ${LIB})
  list(APPEND Tbb_RELEASE_LIBRARY_DIRS ${_TBB_LIBDIR})
endforeach()

foreach(LIB ${Tbb_LIB_COMPONENTS_DEBUG})
  get_filename_component(_TBB_LIBDIR ${LIB} DIRECTORY)
  list(APPEND Tbb_DEBUG_LIBRARIES ${LIB})
  list(APPEND Tbb_DEBUG_LIBRARY_DIRS ${_TBB_LIBDIR})
endforeach()

list(REMOVE_DUPLICATES Tbb_RELEASE_LIBRARY_DIRS)
list(REMOVE_DUPLICATES Tbb_DEBUG_LIBRARY_DIRS)

set(Tbb_LIBRARIES ${Tbb_RELEASE_LIBRARIES})
set(Tbb_LIBRARY_DIRS ${Tbb_RELEASE_LIBRARY_DIRS})
set(Tbb_INCLUDE_DIRS ${Tbb_INCLUDE_DIR})

# Configure imported targets

foreach(COMPONENT ${TBB_FIND_COMPONENTS})
  # Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
  # lib is in use. If win32, we can't mark the import .libs as shared, so
  # these are always marked as UNKNOWN. Otherwise, infer from extension.
  set(TBB_${COMPONENT}_LIB_TYPE UNKNOWN)
  if(TBB_USE_STATIC_LIBS)
    set(TBB_${COMPONENT}_LIB_TYPE STATIC)
  elseif(UNIX)
    get_filename_component(_TBB_${COMPONENT}_EXT ${Tbb_${COMPONENT}_LIBRARY_RELEASE} EXT)
    if(_TBB_${COMPONENT}_EXT STREQUAL ".a")
      set(TBB_${COMPONENT}_LIB_TYPE STATIC)
    elseif(_TBB_${COMPONENT}_EXT STREQUAL ".so" OR
           _TBB_${COMPONENT}_EXT STREQUAL ".dylib")
      set(TBB_${COMPONENT}_LIB_TYPE SHARED)
    endif()
  endif()

  set(Tbb_${COMPONENT}_DEFINITIONS)

  # Add the TBB linking defines if the library is static on WIN32
  if(WIN32)
    if(${COMPONENT} STREQUAL tbb)
      if(Tbb_${COMPONENT}_LIB_TYPE STREQUAL STATIC)
        list(APPEND Tbb_${COMPONENT}_DEFINITIONS __TBB_NO_IMPLICIT_LINKAGE=1)
      endif()
    else() # tbbmalloc
      if(Tbb_${COMPONENT}_LIB_TYPE STREQUAL STATIC)
        list(APPEND Tbb_${COMPONENT}_DEFINITIONS __TBB_MALLOC_NO_IMPLICIT_LINKAGE=1)
      endif()
    endif()
  endif()

  if(NOT TARGET TBB::${COMPONENT})
    add_library(TBB::${COMPONENT} ${TBB_${COMPONENT}_LIB_TYPE} IMPORTED)
    set_target_properties(TBB::${COMPONENT} PROPERTIES
      INTERFACE_COMPILE_OPTIONS "${PC_Tbb_CFLAGS_OTHER}"
      INTERFACE_COMPILE_DEFINITIONS "${Tbb_${COMPONENT}_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${Tbb_INCLUDE_DIR}"
      INTERFACE_LINK_DIRECTORIES "${Tbb_LIBRARY_DIRS}")

    # Standard location
    set_target_properties(TBB::${COMPONENT} PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${Tbb_${COMPONENT}_LIBRARY}")

    # Release location
    if(EXISTS "${Tbb_${COMPONENT}_LIBRARY_RELEASE}")
      set_property(TARGET TBB::${COMPONENT} APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(TBB::${COMPONENT} PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
        IMPORTED_LOCATION_RELEASE "${Tbb_${COMPONENT}_LIBRARY_RELEASE}"
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    endif()

    # Debug location
    if(EXISTS "${Tbb_${COMPONENT}_LIBRARY_DEBUG}")
      set_property(TARGET TBB::${COMPONENT} APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(TBB::${COMPONENT} PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
        IMPORTED_LOCATION_DEBUG "${Tbb_${COMPONENT}_LIBRARY_DEBUG}")
    endif()
  endif()
endforeach()
