# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[.rst:

FindOpenEXR
-----------

Find OpenEXR include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(OpenEXR
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if OpenEXR is not found
    [COMPONENTS <libs>...] # OpenEXR libraries by their canonical name
                           # e.g. "IlmImf" for "libIlmImf"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``OpenEXR::IlmImf``
  The IlmImf library target.
``OpenEXR::IlmImfUtil``
  The IlmImfUtil library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``OpenEXR_FOUND``
  True if the system has the OpenEXR library.
``OpenEXR_VERSION``
  The version of the OpenEXR library which was found.
``OpenEXR_INCLUDE_DIRS``
  Include directories needed to use OpenEXR.
``OpenEXR_RELEASE_LIBRARIES``
  Libraries needed to link to the release version of OpenEXR.
``OpenEXR_RELEASE_LIBRARY_DIRS``
  OpenEXR release library directories.
``OpenEXR_DEBUG_LIBRARIES``
  Libraries needed to link to the debug version of OpenEXR.
``OpenEXR_DEBUG_LIBRARY_DIRS``
  OpenEXR debug library directories.
``OpenEXR_DEFINITIONS``
  Definitions to use when compiling code that uses OpenEXR.
``OpenEXR_{COMPONENT}_FOUND``
  True if the system has the named OpenEXR component.

Deprecated - use [RELEASE|DEBUG] variants:

``OpenEXR_LIBRARIES``
  Libraries needed to link to OpenEXR.
``OpenEXR_LIBRARY_DIRS``
  OpenEXR library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OpenEXR_INCLUDE_DIR``
  The directory containing ``OpenEXR/config-auto.h``.
``OpenEXR_{COMPONENT}_LIBRARY``
  Individual component libraries for OpenEXR. may include target_link_libraries() debug/optimized keywords.
``OpenEXR_{COMPONENT}_LIBRARY_RELEASE``
  Individual component libraries for OpenEXR release
``OpenEXR_{COMPONENT}_LIBRARY_DEBUG``
  Individual component libraries for OpenEXR debug

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``OpenEXR_ROOT``
  Preferred installation prefix.
``OPENEXR_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``OPENEXR_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``OPENEXR_DEBUG_SUFFIX``
  Suffix of the debug version of openexr libs. Defaults to "_d".
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``OPENEXR_USE_STATIC_LIBS``
  Only search for static openexr libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
include(GNUInstallDirs)


mark_as_advanced(
  OpenEXR_INCLUDE_DIR
  OpenEXR_LIBRARY
)

set(_FIND_OPENEXR_ADDITIONAL_OPTIONS "")
if(DISABLE_CMAKE_SEARCH_PATHS)
  set(_FIND_OPENEXR_ADDITIONAL_OPTIONS NO_DEFAULT_PATH)
endif()


# Set _OPENEXR_ROOT based on a user provided root var. Xxx_ROOT and ENV{Xxx_ROOT}
# are prioritised over the legacy capitalized XXX_ROOT variables for matching
# CMake 3.12 behaviour
# @todo  deprecate -D and ENV OPENEXR_ROOT from CMake 3.12
if(OpenEXR_ROOT)
  set(_OPENEXR_ROOT ${OpenEXR_ROOT})
elseif(DEFINED ENV{OpenEXR_ROOT})
  set(_OPENEXR_ROOT $ENV{OpenEXR_ROOT})
elseif(OPENEXR_ROOT)
  set(_OPENEXR_ROOT ${OPENEXR_ROOT})
elseif(DEFINED ENV{OPENEXR_ROOT})
  set(_OPENEXR_ROOT $ENV{OPENEXR_ROOT})
endif()

# Additionally try and use pkconfig to find OpenEXR
if(USE_PKGCONFIG)
  if(NOT DEFINED PKG_CONFIG_FOUND)
    find_package(PkgConfig)
  endif()
  pkg_check_modules(PC_OpenEXR QUIET OpenEXR)
endif()

# ------------------------------------------------------------------------
#  Search for OpenEXR include DIR
# ------------------------------------------------------------------------

set(_OPENEXR_INCLUDE_SEARCH_DIRS "")
list(APPEND _OPENEXR_INCLUDE_SEARCH_DIRS
  ${OPENEXR_INCLUDEDIR}
  ${_OPENEXR_ROOT}
  ${PC_OpenEXR_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard OpenEXR header file.
find_path(OpenEXR_INCLUDE_DIR OpenEXRConfig.h
  ${_FIND_OPENEXR_ADDITIONAL_OPTIONS}
  PATHS ${_OPENEXR_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR}/OpenEXR include/OpenEXR OpenEXR
)

if(EXISTS "${OpenEXR_INCLUDE_DIR}/OpenEXRConfig.h")
  # Get the EXR version information from the config header
  file(STRINGS "${OpenEXR_INCLUDE_DIR}/OpenEXRConfig.h"
    _openexr_version_major_string REGEX "#define OPENEXR_VERSION_MAJOR "
  )
  string(REGEX REPLACE "#define OPENEXR_VERSION_MAJOR" ""
    _openexr_version_major_string "${_openexr_version_major_string}"
  )
  string(STRIP "${_openexr_version_major_string}" OpenEXR_VERSION_MAJOR)

  file(STRINGS "${OpenEXR_INCLUDE_DIR}/OpenEXRConfig.h"
     _openexr_version_minor_string REGEX "#define OPENEXR_VERSION_MINOR "
  )
  string(REGEX REPLACE "#define OPENEXR_VERSION_MINOR" ""
    _openexr_version_minor_string "${_openexr_version_minor_string}"
  )
  string(STRIP "${_openexr_version_minor_string}" OpenEXR_VERSION_MINOR)

  unset(_openexr_version_major_string)
  unset(_openexr_version_minor_string)

  set(OpenEXR_VERSION ${OpenEXR_VERSION_MAJOR}.${OpenEXR_VERSION_MINOR})
endif()

if(${OpenEXR_VERSION} VERSION_GREATER_EQUAL 3.0)
  set(_OPENEXR_COMPONENT_LIST OpenEXR OpenEXRUtil Iex IlmThread)
else()
  set(_OPENEXR_COMPONENT_LIST IlmImf IlmImfUtil)
endif()

if(OpenEXR_FIND_COMPONENTS)
  set(OPENEXR_COMPONENTS_PROVIDED TRUE)
  set(_IGNORED_COMPONENTS "")
  foreach(COMPONENT ${OpenEXR_FIND_COMPONENTS})
    if(NOT ${COMPONENT} IN_LIST _OPENEXR_COMPONENT_LIST)
      list(APPEND _IGNORED_COMPONENTS ${COMPONENT})
    endif()
  endforeach()

  if(_IGNORED_COMPONENTS)
    message(STATUS "Ignoring unknown components of OpenEXR:")
    foreach(COMPONENT ${_IGNORED_COMPONENTS})
      message(STATUS "  ${COMPONENT}")
    endforeach()
    list(REMOVE_ITEM OpenEXR_FIND_COMPONENTS ${_IGNORED_COMPONENTS})
  endif()
else()
  set(OPENEXR_COMPONENTS_PROVIDED FALSE)
  set(OpenEXR_FIND_COMPONENTS ${_OPENEXR_COMPONENT_LIST})
endif()


# ------------------------------------------------------------------------
#  Search for OPENEXR lib DIR
# ------------------------------------------------------------------------

if(NOT DEFINED OPENEXR_DEBUG_SUFFIX)
  set(OPENEXR_DEBUG_SUFFIX _d)
endif()

set(_OPENEXR_LIBRARYDIR_SEARCH_DIRS "")

# Append to _OPENEXR_LIBRARYDIR_SEARCH_DIRS in priority order

list(APPEND _OPENEXR_LIBRARYDIR_SEARCH_DIRS
  ${OPENEXR_LIBRARYDIR}
  ${_OPENEXR_ROOT}
  ${PC_OpenEXR_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

set(OpenEXR_LIB_COMPONENTS "")
list(APPEND OPENEXR_BUILD_TYPES RELEASE DEBUG)

foreach(COMPONENT ${OpenEXR_FIND_COMPONENTS})
  foreach(BUILD_TYPE ${OPENEXR_BUILD_TYPES})

    set(_TMP_SUFFIX "")
    if(BUILD_TYPE STREQUAL DEBUG)
      set(_TMP_SUFFIX ${OPENEXR_DEBUG_SUFFIX})
    endif()

    set(_OpenEXR_Version_Suffix "-${OpenEXR_VERSION_MAJOR}_${OpenEXR_VERSION_MINOR}")
    set(_OPENEXR_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

    if(WIN32)
      if(OPENEXR_USE_STATIC_LIBS)
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${_TMP_SUFFIX}.lib")
      endif()
      list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "${_OpenEXR_Version_Suffix}${_TMP_SUFFIX}.lib")
    else()
      if(OPENEXR_USE_STATIC_LIBS)
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${_TMP_SUFFIX}.a")
      else()
        if(APPLE)
          list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "${_OpenEXR_Version_Suffix}${_TMP_SUFFIX}.dylib")
        else()
          list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "${_OpenEXR_Version_Suffix}${_TMP_SUFFIX}.so")
        endif()
      endif()
      list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "${_OpenEXR_Version_Suffix}${_TMP_SUFFIX}.a")
    endif()

    # Find the lib
    find_library(OpenEXR_${COMPONENT}_LIBRARY_${BUILD_TYPE} ${COMPONENT}
      ${_FIND_OPENEXR_ADDITIONAL_OPTIONS}
      PATHS ${_OPENEXR_LIBRARYDIR_SEARCH_DIRS}
      PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
    )

    if(EXISTS ${OpenEXR_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
      list(APPEND OpenEXR_LIB_COMPONENTS ${OpenEXR_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
      list(APPEND OpenEXR_LIB_COMPONENTS_${BUILD_TYPE} ${OpenEXR_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
    endif()

    # Reset library suffix
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_OPENEXR_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
    unset(_OPENEXR_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
    unset(_OpenEXR_Version_Suffix)
    unset(_TMP_SUFFIX)
  endforeach()

  if(OpenEXR_${COMPONENT}_LIBRARY_DEBUG AND OpenEXR_${COMPONENT}_LIBRARY_RELEASE)
    # if the generator is multi-config or if CMAKE_BUILD_TYPE is set for
    # single-config generators, set optimized and debug libraries
    get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_isMultiConfig OR CMAKE_BUILD_TYPE)
      set(OpenEXR_${COMPONENT}_LIBRARY optimized ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE} debug ${OpenEXR_${COMPONENT}_LIBRARY_DEBUG})
    else()
      # For single-config generators where CMAKE_BUILD_TYPE has no value,
      # just use the release libraries
      set(OpenEXR_${COMPONENT}_LIBRARY ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE})
    endif()
    # FIXME: This probably should be set for both cases
    set(OpenEXR_${COMPONENT}_LIBRARIES optimized ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE} debug ${OpenEXR_${COMPONENT}_LIBRARY_DEBUG})
  endif()

  # if only the release version was found, set the debug variable also to the release version
  if(OpenEXR_${COMPONENT}_LIBRARY_RELEASE AND NOT OpenEXR_${COMPONENT}_LIBRARY_DEBUG)
    set(OpenEXR_${COMPONENT}_LIBRARY_DEBUG ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE})
    set(OpenEXR_${COMPONENT}_LIBRARY       ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE})
    set(OpenEXR_${COMPONENT}_LIBRARIES     ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE})
  endif()

  # if only the debug version was found, set the release variable also to the debug version
  if(OpenEXR_${COMPONENT}_LIBRARY_DEBUG AND NOT OpenEXR_${COMPONENT}_LIBRARY_RELEASE)
    set(OpenEXR_${COMPONENT}_LIBRARY_RELEASE ${OpenEXR_${COMPONENT}_LIBRARY_DEBUG})
    set(OpenEXR_${COMPONENT}_LIBRARY         ${OpenEXR_${COMPONENT}_LIBRARY_DEBUG})
    set(OpenEXR_${COMPONENT}_LIBRARIES       ${OpenEXR_${COMPONENT}_LIBRARY_DEBUG})
  endif()

  # If the debug & release library ends up being the same, omit the keywords
  if("${OpenEXR_${COMPONENT}_LIBRARY_RELEASE}" STREQUAL "${OpenEXR_${COMPONENT}_LIBRARY_DEBUG}")
    set(OpenEXR_${COMPONENT}_LIBRARY   ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE} )
    set(OpenEXR_${COMPONENT}_LIBRARIES ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE} )
  endif()

  if(OpenEXR_${COMPONENT}_LIBRARY)
    set(OpenEXR_${COMPONENT}_FOUND TRUE)
  else()
    set(OpenEXR_${COMPONENT}_FOUND FALSE)
  endif()
endforeach()

# ------------------------------------------------------------------------
#  Cache and set OPENEXR_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenEXR
  FOUND_VAR OpenEXR_FOUND
  REQUIRED_VARS
    OpenEXR_INCLUDE_DIR
    OpenEXR_LIB_COMPONENTS
  VERSION_VAR OpenEXR_VERSION
  HANDLE_COMPONENTS
)

if(NOT OpenEXR_FOUND)
  if(OpenEXR_FIND_REQUIRED)
    message(FATAL_ERROR "Unable to find OpenEXR")
  endif()
  return()
endif()

# Partition release/debug lib vars

set(OpenEXR_RELEASE_LIBRARIES "")
set(OpenEXR_RELEASE_LIBRARY_DIRS "")
set(OpenEXR_DEBUG_LIBRARIES "")
set(OpenEXR_DEBUG_LIBRARY_DIRS "")
foreach(LIB ${OpenEXR_LIB_COMPONENTS_RELEASE})
  get_filename_component(_EXR_LIBDIR ${LIB} DIRECTORY)
  list(APPEND OpenEXR_RELEASE_LIBRARIES ${LIB})
  list(APPEND OpenEXR_RELEASE_LIBRARY_DIRS ${_EXR_LIBDIR})
endforeach()

foreach(LIB ${OpenEXR_LIB_COMPONENTS_DEBUG})
  get_filename_component(_EXR_LIBDIR ${LIB} DIRECTORY)
  list(APPEND OpenEXR_DEBUG_LIBRARIES ${LIB})
  list(APPEND OpenEXR_DEBUG_LIBRARY_DIRS ${_EXR_LIBDIR})
endforeach()

list(REMOVE_DUPLICATES OpenEXR_RELEASE_LIBRARY_DIRS)
list(REMOVE_DUPLICATES OpenEXR_DEBUG_LIBRARY_DIRS)

set(OpenEXR_LIBRARIES ${OpenEXR_RELEASE_LIBRARIES})
set(OpenEXR_LIBRARY_DIRS ${OpenEXR_RELEASE_LIBRARY_DIRS})

# We have to add both include and include/OpenEXR to the include
# path in case OpenEXR and IlmBase are installed separately.
#
# Make sure we get the absolute path to avoid issues where
# /usr/include/OpenEXR/../ is picked up and passed to gcc from cmake
# which won't correctly compute /usr/include as an implicit system
# dir if the path is relative:
#
# https://github.com/AcademySoftwareFoundation/openvdb/issues/632
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70129

set(_OpenEXR_Parent_Dir "")
get_filename_component(_OpenEXR_Parent_Dir
  ${OpenEXR_INCLUDE_DIR}/../ ABSOLUTE)

set(OpenEXR_INCLUDE_DIRS)
list(APPEND OpenEXR_INCLUDE_DIRS
  ${_OpenEXR_Parent_Dir}
  ${OpenEXR_INCLUDE_DIR}
)
unset(_OpenEXR_Parent_Dir)

# Configure imported target

foreach(COMPONENT ${OpenEXR_FIND_COMPONENTS})
  # Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
  # lib is in use. If win32, we can't mark the import .libs as shared, so
  # these are always marked as UNKNOWN. Otherwise, infer from extension.
  set(OpenEXR_${COMPONENT}_LIB_TYPE UNKNOWN)
  if(OPENEXR_USE_STATIC_LIBS)
    set(OpenEXR_${COMPONENT}_LIB_TYPE STATIC)
  elseif(UNIX)
    get_filename_component(_OpenEXR_${COMPONENT}_EXT ${OpenEXR_${COMPONENT}_LIBRARY_RELEASE} EXT)
    if(${_OpenEXR_${COMPONENT}_EXT} STREQUAL ".a")
      set(OpenEXR_${COMPONENT}_LIB_TYPE STATIC)
    elseif(${_OpenEXR_${COMPONENT}_EXT} STREQUAL ".so" OR
           ${_OpenEXR_${COMPONENT}_EXT} STREQUAL ".dylib")
      set(OpenEXR_${COMPONENT}_LIB_TYPE SHARED)
    endif()
  endif()

  set(OpenEXR_${COMPONENT}_DEFINITIONS)

  # Add the OPENEXR_DLL define if the library is not static on WIN32
  if(WIN32)
    if(NOT OpenEXR_${COMPONENT}_LIB_TYPE STREQUAL STATIC)
      list(APPEND OpenEXR_${COMPONENT}_DEFINITIONS OPENEXR_DLL)
    endif()
  endif()

  if(NOT TARGET OpenEXR::${COMPONENT})
    add_library(OpenEXR::${COMPONENT} ${OpenEXR_${COMPONENT}_LIB_TYPE} IMPORTED)
    set_target_properties(OpenEXR::${COMPONENT} PROPERTIES
      INTERFACE_COMPILE_OPTIONS "${PC_OpenEXR_CFLAGS_OTHER}"
      INTERFACE_COMPILE_DEFINITIONS "${OpenEXR_${COMPONENT}_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${OpenEXR_INCLUDE_DIRS}")

    # Standard location
    set_target_properties(OpenEXR::${COMPONENT} PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${OpenEXR_${COMPONENT}_LIBRARY}")

    # Release location
    if(EXISTS "${OpenEXR_${COMPONENT}_LIBRARY_RELEASE}")
      set_property(TARGET OpenEXR::${COMPONENT} APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(OpenEXR::${COMPONENT} PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
        IMPORTED_LOCATION_RELEASE "${OpenEXR_${COMPONENT}_LIBRARY_RELEASE}")
    endif()

    # Debug location
    if(EXISTS "${OpenEXR_${COMPONENT}_LIBRARY_DEBUG}")
      set_property(TARGET OpenEXR::${COMPONENT} APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(OpenEXR::${COMPONENT} PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
        IMPORTED_LOCATION_DEBUG "${OpenEXR_${COMPONENT}_LIBRARY_DEBUG}")
    endif()
  endif()
endforeach()

