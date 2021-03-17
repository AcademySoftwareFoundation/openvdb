# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[.rst:

FindIlmBase
-----------

Find IlmBase include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(IlmBase
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if IlmBase is not found
    [COMPONENTS <libs>...] # IlmBase libraries by their canonical name
                           # e.g. "Half" for "libHalf"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``IlmBase::Half``
  The Half library target.
``IlmBase::Iex``
  The Iex library target.
``IlmBase::IexMath``
  The IexMath library target.
``IlmBase::IlmThread``
  The IlmThread library target.
``IlmBase::Imath``
  The Imath library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``IlmBase_FOUND``
  True if the system has the IlmBase library.
``IlmBase_VERSION``
  The version of the IlmBase library which was found.
``IlmBase_INCLUDE_DIRS``
  Include directories needed to use IlmBase.
``IlmBase_RELEASE_LIBRARIES``
  Libraries needed to link to the release version of IlmBase.
``IlmBase_RELEASE_LIBRARY_DIRS``
  IlmBase release library directories.
``IlmBase_DEBUG_LIBRARIES``
  Libraries needed to link to the debug version of IlmBase.
``IlmBase_DEBUG_LIBRARY_DIRS``
  IlmBase debug library directories.
``IlmBase_{COMPONENT}_FOUND``
  True if the system has the named IlmBase component.

Deprecated - use [RELEASE|DEBUG] variants:

``IlmBase_LIBRARIES``
  Libraries needed to link to IlmBase.
``IlmBase_LIBRARY_DIRS``
  IlmBase library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``IlmBase_INCLUDE_DIR``
  The directory containing ``IlmBase/config-auto.h``.
``IlmBase_{COMPONENT}_LIBRARY``
  Individual component libraries for IlmBase. may include target_link_libraries() debug/optimized keywords.
``IlmBase_{COMPONENT}_LIBRARY_RELEASE``
  Individual component libraries for IlmBase release
``IlmBase_{COMPONENT}_LIBRARY_DEBUG``
  Individual component libraries for IlmBase debug

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``IlmBase_ROOT``
  Preferred installation prefix.
``ILMBASE_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``ILMBASE_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``ILMBASE_DEBUG_SUFFIX``
  Suffix of the debug version of ilmbase libs. Defaults to "_d".
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``ILMBASE_USE_STATIC_LIBS``
  Only search for static ilmbase libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
include(GNUInstallDirs)


mark_as_advanced(
  IlmBase_INCLUDE_DIR
  IlmBase_LIBRARY
)

set(_FIND_ILMBASE_ADDITIONAL_OPTIONS "")
if(DISABLE_CMAKE_SEARCH_PATHS)
  set(_FIND_ILMBASE_ADDITIONAL_OPTIONS NO_DEFAULT_PATH)
endif()

set(_ILMBASE_COMPONENT_LIST
  Half
  Iex
  IexMath
  IlmThread
  Imath
)

if(IlmBase_FIND_COMPONENTS)
  set(ILMBASE_COMPONENTS_PROVIDED TRUE)
  set(_IGNORED_COMPONENTS "")
  foreach(COMPONENT ${IlmBase_FIND_COMPONENTS})
    if(NOT ${COMPONENT} IN_LIST _ILMBASE_COMPONENT_LIST)
      list(APPEND _IGNORED_COMPONENTS ${COMPONENT})
    endif()
  endforeach()

  if(_IGNORED_COMPONENTS)
    message(STATUS "Ignoring unknown components of IlmBase:")
    foreach(COMPONENT ${_IGNORED_COMPONENTS})
      message(STATUS "  ${COMPONENT}")
    endforeach()
    list(REMOVE_ITEM IlmBase_FIND_COMPONENTS ${_IGNORED_COMPONENTS})
  endif()
else()
  set(ILMBASE_COMPONENTS_PROVIDED FALSE)
  set(IlmBase_FIND_COMPONENTS ${_ILMBASE_COMPONENT_LIST})
endif()

# Set _ILMBASE_ROOT based on a user provided root var. Xxx_ROOT and ENV{Xxx_ROOT}
# are prioritised over the legacy capitalized XXX_ROOT variables for matching
# CMake 3.12 behaviour
# @todo  deprecate -D and ENV ILMBASE_ROOT from CMake 3.12
if(IlmBase_ROOT)
  set(_ILMBASE_ROOT ${IlmBase_ROOT})
elseif(DEFINED ENV{IlmBase_ROOT})
  set(_ILMBASE_ROOT $ENV{IlmBase_ROOT})
elseif(ILMBASE_ROOT)
  set(_ILMBASE_ROOT ${ILMBASE_ROOT})
elseif(DEFINED ENV{ILMBASE_ROOT})
  set(_ILMBASE_ROOT $ENV{ILMBASE_ROOT})
endif()

# Additionally try and use pkconfig to find IlmBase
if(USE_PKGCONFIG)
  if(NOT DEFINED PKG_CONFIG_FOUND)
    find_package(PkgConfig)
  endif()
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_IlmBase QUIET ilmbase)
  endif()
endif()

# ------------------------------------------------------------------------
#  Search for IlmBase include DIR
# ------------------------------------------------------------------------

set(_ILMBASE_INCLUDE_SEARCH_DIRS "")
list(APPEND _ILMBASE_INCLUDE_SEARCH_DIRS
  ${ILMBASE_INCLUDEDIR}
  ${_ILMBASE_ROOT}
  ${PC_IlmBase_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard IlmBase header file.
find_path(IlmBase_INCLUDE_DIR IlmBaseConfig.h
  ${_FIND_ILMBASE_ADDITIONAL_OPTIONS}
  PATHS ${_ILMBASE_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR}/OpenEXR include/OpenEXR OpenEXR
)

if(EXISTS "${IlmBase_INCLUDE_DIR}/IlmBaseConfig.h")
  # Get the ILMBASE version information from the config header
  file(STRINGS "${IlmBase_INCLUDE_DIR}/IlmBaseConfig.h"
    _ilmbase_version_major_string REGEX "#define ILMBASE_VERSION_MAJOR "
  )
  string(REGEX REPLACE "#define ILMBASE_VERSION_MAJOR" ""
    _ilmbase_version_major_string "${_ilmbase_version_major_string}"
  )
  string(STRIP "${_ilmbase_version_major_string}" IlmBase_VERSION_MAJOR)

  file(STRINGS "${IlmBase_INCLUDE_DIR}/IlmBaseConfig.h"
     _ilmbase_version_minor_string REGEX "#define ILMBASE_VERSION_MINOR "
  )
  string(REGEX REPLACE "#define ILMBASE_VERSION_MINOR" ""
    _ilmbase_version_minor_string "${_ilmbase_version_minor_string}"
  )
  string(STRIP "${_ilmbase_version_minor_string}" IlmBase_VERSION_MINOR)

  unset(_ilmbase_version_major_string)
  unset(_ilmbase_version_minor_string)

  set(IlmBase_VERSION ${IlmBase_VERSION_MAJOR}.${IlmBase_VERSION_MINOR})
endif()

# ------------------------------------------------------------------------
#  Search for ILMBASE lib DIR
# ------------------------------------------------------------------------

if(NOT DEFINED ILMBASE_DEBUG_SUFFIX)
  set(ILMBASE_DEBUG_SUFFIX _d)
endif()

set(_ILMBASE_LIBRARYDIR_SEARCH_DIRS "")

# Append to _ILMBASE_LIBRARYDIR_SEARCH_DIRS in priority order

list(APPEND _ILMBASE_LIBRARYDIR_SEARCH_DIRS
  ${ILMBASE_LIBRARYDIR}
  ${_ILMBASE_ROOT}
  ${PC_IlmBase_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

set(IlmBase_LIB_COMPONENTS "")
list(APPEND ILM_BUILD_TYPES RELEASE DEBUG)

foreach(COMPONENT ${IlmBase_FIND_COMPONENTS})
  foreach(BUILD_TYPE ${ILM_BUILD_TYPES})

    set(_TMP_SUFFIX "")
    if(BUILD_TYPE STREQUAL DEBUG)
      set(_TMP_SUFFIX ${ILMBASE_DEBUG_SUFFIX})
    endif()

    set(_IlmBase_Version_Suffix "-${IlmBase_VERSION_MAJOR}_${IlmBase_VERSION_MINOR}")
    set(_ILMBASE_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

    if(WIN32)
      if(ILMBASE_USE_STATIC_LIBS)
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${_TMP_SUFFIX}.lib")
      endif()
      list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "${_IlmBase_Version_Suffix}${_TMP_SUFFIX}.lib")
    else()
      if(ILMBASE_USE_STATIC_LIBS)
        set(CMAKE_FIND_LIBRARY_SUFFIXES "${_TMP_SUFFIX}.a")
      else()
        if(APPLE)
          list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "${_IlmBase_Version_Suffix}${_TMP_SUFFIX}.dylib")
        else()
          list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "${_IlmBase_Version_Suffix}${_TMP_SUFFIX}.so")
        endif()
      endif()
      list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES "${_IlmBase_Version_Suffix}${_TMP_SUFFIX}.a")
    endif()

    # Find the lib
    find_library(IlmBase_${COMPONENT}_LIBRARY_${BUILD_TYPE} ${COMPONENT}
      ${_FIND_ILMBASE_ADDITIONAL_OPTIONS}
      PATHS ${_ILMBASE_LIBRARYDIR_SEARCH_DIRS}
      PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
    )

    if(EXISTS ${IlmBase_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
      list(APPEND IlmBase_LIB_COMPONENTS ${IlmBase_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
      list(APPEND IlmBase_LIB_COMPONENTS_${BUILD_TYPE} ${IlmBase_${COMPONENT}_LIBRARY_${BUILD_TYPE}})
    endif()

    # Reset library suffix
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ILMBASE_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
    unset(_ILMBASE_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
    unset(_IlmBase_Version_Suffix)
    unset(_TMP_SUFFIX)
  endforeach()

  if(IlmBase_${COMPONENT}_LIBRARY_DEBUG AND IlmBase_${COMPONENT}_LIBRARY_RELEASE)
    # if the generator is multi-config or if CMAKE_BUILD_TYPE is set for
    # single-config generators, set optimized and debug libraries
    get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_isMultiConfig OR CMAKE_BUILD_TYPE)
      set(IlmBase_${COMPONENT}_LIBRARY optimized ${IlmBase_${COMPONENT}_LIBRARY_RELEASE} debug ${IlmBase_${COMPONENT}_LIBRARY_DEBUG})
    else()
      # For single-config generators where CMAKE_BUILD_TYPE has no value,
      # just use the release libraries
      set(IlmBase_${COMPONENT}_LIBRARY ${IlmBase_${COMPONENT}_LIBRARY_RELEASE})
    endif()
    # FIXME: This probably should be set for both cases
    set(IlmBase_${COMPONENT}_LIBRARIES optimized ${IlmBase_${COMPONENT}_LIBRARY_RELEASE} debug ${IlmBase_${COMPONENT}_LIBRARY_DEBUG})
  endif()

  # if only the release version was found, set the debug variable also to the release version
  if(IlmBase_${COMPONENT}_LIBRARY_RELEASE AND NOT IlmBase_${COMPONENT}_LIBRARY_DEBUG)
    set(IlmBase_${COMPONENT}_LIBRARY_DEBUG ${IlmBase_${COMPONENT}_LIBRARY_RELEASE})
    set(IlmBase_${COMPONENT}_LIBRARY       ${IlmBase_${COMPONENT}_LIBRARY_RELEASE})
    set(IlmBase_${COMPONENT}_LIBRARIES     ${IlmBase_${COMPONENT}_LIBRARY_RELEASE})
  endif()

  # if only the debug version was found, set the release variable also to the debug version
  if(IlmBase_${COMPONENT}_LIBRARY_DEBUG AND NOT IlmBase_${COMPONENT}_LIBRARY_RELEASE)
    set(IlmBase_${COMPONENT}_LIBRARY_RELEASE ${IlmBase_${COMPONENT}_LIBRARY_DEBUG})
    set(IlmBase_${COMPONENT}_LIBRARY         ${IlmBase_${COMPONENT}_LIBRARY_DEBUG})
    set(IlmBase_${COMPONENT}_LIBRARIES       ${IlmBase_${COMPONENT}_LIBRARY_DEBUG})
  endif()

  # If the debug & release library ends up being the same, omit the keywords
  if("${IlmBase_${COMPONENT}_LIBRARY_RELEASE}" STREQUAL "${IlmBase_${COMPONENT}_LIBRARY_DEBUG}")
    set(IlmBase_${COMPONENT}_LIBRARY   ${IlmBase_${COMPONENT}_LIBRARY_RELEASE} )
    set(IlmBase_${COMPONENT}_LIBRARIES ${IlmBase_${COMPONENT}_LIBRARY_RELEASE} )
  endif()

  if(IlmBase_${COMPONENT}_LIBRARY)
    set(IlmBase_${COMPONENT}_FOUND TRUE)
  else()
    set(IlmBase_${COMPONENT}_FOUND FALSE)
  endif()
endforeach()

# ------------------------------------------------------------------------
#  Cache and set ILMBASE_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IlmBase
  FOUND_VAR IlmBase_FOUND
  REQUIRED_VARS
    IlmBase_INCLUDE_DIR
    IlmBase_LIB_COMPONENTS
  VERSION_VAR IlmBase_VERSION
  HANDLE_COMPONENTS
)

if(NOT IlmBase_FOUND)
  if(IlmBase_FIND_REQUIRED)
    message(FATAL_ERROR "Unable to find IlmBase")
  endif()
  return()
endif()

# Partition release/debug lib vars

set(IlmBase_RELEASE_LIBRARIES "")
set(IlmBase_RELEASE_LIBRARY_DIRS "")
set(IlmBase_DEBUG_LIBRARIES "")
set(IlmBase_DEBUG_LIBRARY_DIRS "")
foreach(LIB ${IlmBase_LIB_COMPONENTS_RELEASE})
  get_filename_component(_ILM_LIBDIR ${LIB} DIRECTORY)
  list(APPEND IlmBase_RELEASE_LIBRARIES ${LIB})
  list(APPEND IlmBase_RELEASE_LIBRARY_DIRS ${_ILM_LIBDIR})
endforeach()

foreach(LIB ${IlmBase_LIB_COMPONENTS_DEBUG})
  get_filename_component(_ILM_LIBDIR ${LIB} DIRECTORY)
  list(APPEND IlmBase_DEBUG_LIBRARIES ${LIB})
  list(APPEND IlmBase_DEBUG_LIBRARY_DIRS ${_ILM_LIBDIR})
endforeach()

list(REMOVE_DUPLICATES IlmBase_RELEASE_LIBRARY_DIRS)
list(REMOVE_DUPLICATES IlmBase_DEBUG_LIBRARY_DIRS)

set(IlmBase_LIBRARIES ${IlmBase_RELEASE_LIBRARIES})
set(IlmBase_LIBRARY_DIRS ${IlmBase_RELEASE_LIBRARY_DIRS})

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

set(_IlmBase_Parent_Dir "")
get_filename_component(_IlmBase_Parent_Dir
  ${IlmBase_INCLUDE_DIR}/../ ABSOLUTE)

set(IlmBase_INCLUDE_DIRS)
list(APPEND IlmBase_INCLUDE_DIRS
  ${_IlmBase_Parent_Dir}
  ${IlmBase_INCLUDE_DIR}
)
unset(_IlmBase_Parent_Dir)

# Configure imported targets

foreach(COMPONENT ${IlmBase_FIND_COMPONENTS})
  # Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
  # lib is in use. If win32, we can't mark the import .libs as shared, so
  # these are always marked as UNKNOWN. Otherwise, infer from extension.
  set(ILMBASE_${COMPONENT}_LIB_TYPE UNKNOWN)
  if(ILMBASE_USE_STATIC_LIBS)
    set(ILMBASE_${COMPONENT}_LIB_TYPE STATIC)
  elseif(UNIX)
    get_filename_component(_ILMBASE_${COMPONENT}_EXT ${IlmBase_${COMPONENT}_LIBRARY_RELEASE} EXT)
    if(${_ILMBASE_${COMPONENT}_EXT} STREQUAL ".a")
      set(ILMBASE_${COMPONENT}_LIB_TYPE STATIC)
    elseif(${_ILMBASE_${COMPONENT}_EXT} STREQUAL ".so" OR
           ${_ILMBASE_${COMPONENT}_EXT} STREQUAL ".dylib")
      set(ILMBASE_${COMPONENT}_LIB_TYPE SHARED)
    endif()
  endif()

  set(IlmBase_${COMPONENT}_DEFINITIONS)

  # Add the OPENEXR_DLL define if the library is not static on WIN32
  if(WIN32)
    if(NOT ILMBASE_${COMPONENT}_LIB_TYPE STREQUAL STATIC)
      list(APPEND IlmBase_${COMPONENT}_DEFINITIONS OPENEXR_DLL)
    endif()
  endif()

  if(NOT TARGET IlmBase::${COMPONENT})
    add_library(IlmBase::${COMPONENT} ${ILMBASE_${COMPONENT}_LIB_TYPE} IMPORTED)
    set_target_properties(IlmBase::${COMPONENT} PROPERTIES
      INTERFACE_COMPILE_OPTIONS "${PC_IlmBase_CFLAGS_OTHER}"
      INTERFACE_COMPILE_DEFINITIONS "${IlmBase_${COMPONENT}_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${IlmBase_INCLUDE_DIRS}")

    # Standard location
    set_target_properties(IlmBase::${COMPONENT} PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${IlmBase_${COMPONENT}_LIBRARY}")

    # Release location
    if(EXISTS "${IlmBase_${COMPONENT}_LIBRARY_RELEASE}")
      set_property(TARGET IlmBase::${COMPONENT} APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(IlmBase::${COMPONENT} PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
        IMPORTED_LOCATION_RELEASE "${IlmBase_${COMPONENT}_LIBRARY_RELEASE}")
    endif()

    # Debug location
    if(EXISTS "${IlmBase_${COMPONENT}_LIBRARY_DEBUG}")
      set_property(TARGET IlmBase::${COMPONENT} APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(IlmBase::${COMPONENT} PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
        IMPORTED_LOCATION_DEBUG "${IlmBase_${COMPONENT}_LIBRARY_DEBUG}")
    endif()
  endif()
endforeach()
