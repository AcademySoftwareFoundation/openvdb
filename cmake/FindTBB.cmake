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
``Tbb_VERSION``
  The version of the Tbb library which was found.
``Tbb_INCLUDE_DIRS``
  Include directories needed to use Tbb.
``Tbb_LIBRARIES``
  Libraries needed to link to Tbb.
``Tbb_LIBRARY_DIRS``
  Tbb library directories.
``TBB_{COMPONENT}_FOUND``
  True if the system has the named TBB component.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Tbb_INCLUDE_DIR``
  The directory containing ``tbb/tbb_stddef.h``.
``Tbb_{COMPONENT}_LIBRARY``
  Individual component libraries for Tbb

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
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``TBB_USE_STATIC_LIBS``
  Only search for static tbb libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.3)
include(GNUInstallDirs)

# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

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
  pkg_check_modules(PC_Tbb QUIET tbb)
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

# Look for a standard tbb header file.
find_path(Tbb_INCLUDE_DIR tbb/tbb_stddef.h
  ${_FIND_TBB_ADDITIONAL_OPTIONS}
  PATHS ${_TBB_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES ${CMAKE_INSTALL_INCLUDEDIR} include
)

if(EXISTS "${Tbb_INCLUDE_DIR}/tbb/tbb_stddef.h")
  file(STRINGS "${Tbb_INCLUDE_DIR}/tbb/tbb_stddef.h"
    _tbb_version_major_string REGEX "#define TBB_VERSION_MAJOR "
  )
  string(REGEX REPLACE "#define TBB_VERSION_MAJOR" ""
    _tbb_version_major_string "${_tbb_version_major_string}"
  )
  string(STRIP "${_tbb_version_major_string}" Tbb_VERSION_MAJOR)

  file(STRINGS "${Tbb_INCLUDE_DIR}/tbb/tbb_stddef.h"
     _tbb_version_minor_string REGEX "#define TBB_VERSION_MINOR "
  )
  string(REGEX REPLACE "#define TBB_VERSION_MINOR" ""
    _tbb_version_minor_string "${_tbb_version_minor_string}"
  )
  string(STRIP "${_tbb_version_minor_string}" Tbb_VERSION_MINOR)

  unset(_tbb_version_major_string)
  unset(_tbb_version_minor_string)

  set(Tbb_VERSION ${Tbb_VERSION_MAJOR}.${Tbb_VERSION_MINOR})
endif()

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

set(_TBB_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(WIN32)
  if(TBB_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
  endif()
else()
  if(TBB_USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  endif()
endif()

set(Tbb_LIB_COMPONENTS "")

foreach(COMPONENT ${TBB_FIND_COMPONENTS})
  find_library(Tbb_${COMPONENT}_LIBRARY ${COMPONENT}
    ${_FIND_TBB_ADDITIONAL_OPTIONS}
    PATHS ${_TBB_LIBRARYDIR_SEARCH_DIRS}
    PATH_SUFFIXES ${CMAKE_INSTALL_LIBDIR} lib64 lib
  )

  # On Unix, TBB sometimes uses linker scripts instead of symlinks, so parse the linker script
  # and correct the library name if so
  if(UNIX AND EXISTS ${Tbb_${COMPONENT}_LIBRARY})
    # Ignore files where the first four bytes equals the ELF magic number
    file(READ ${Tbb_${COMPONENT}_LIBRARY} Tbb_${COMPONENT}_HEX OFFSET 0 LIMIT 4 HEX)
    if(NOT ${Tbb_${COMPONENT}_HEX} STREQUAL "7f454c46")
      # Read the first 1024 bytes of the library and match against an "INPUT (file)" regex
      file(READ ${Tbb_${COMPONENT}_LIBRARY} Tbb_${COMPONENT}_ASCII OFFSET 0 LIMIT 1024)
      if("${Tbb_${COMPONENT}_ASCII}" MATCHES "INPUT \\(([^(]+)\\)")
        # Extract the directory and apply the matched text (in brackets)
        get_filename_component(Tbb_${COMPONENT}_DIR "${Tbb_${COMPONENT}_LIBRARY}" DIRECTORY)
        set(Tbb_${COMPONENT}_LIBRARY "${Tbb_${COMPONENT}_DIR}/${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()

  list(APPEND Tbb_LIB_COMPONENTS ${Tbb_${COMPONENT}_LIBRARY})

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

if(TBB_FOUND)
  set(Tbb_LIBRARIES ${Tbb_LIB_COMPONENTS})
  set(Tbb_INCLUDE_DIRS ${Tbb_INCLUDE_DIR})

  set(Tbb_LIBRARY_DIRS "")
  foreach(LIB ${Tbb_LIB_COMPONENTS})
    get_filename_component(_TBB_LIBDIR ${LIB} DIRECTORY)
    list(APPEND Tbb_LIBRARY_DIRS ${_TBB_LIBDIR})
  endforeach()
  list(REMOVE_DUPLICATES Tbb_LIBRARY_DIRS)

  # Configure imported targets

  foreach(COMPONENT ${TBB_FIND_COMPONENTS})
    # Configure lib type. If XXX_USE_STATIC_LIBS, we always assume a static
    # lib is in use. If win32, we can't mark the import .libs as shared, so
    # these are always marked as UNKNOWN. Otherwise, infer from extension.
    set(TBB_${COMPONENT}_LIB_TYPE UNKNOWN)
    if(TBB_USE_STATIC_LIBS)
      set(TBB_${COMPONENT}_LIB_TYPE STATIC)
    elseif(UNIX)
      get_filename_component(_TBB_${COMPONENT}_EXT ${Tbb_${COMPONENT}_LIBRARY} EXT)
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
        IMPORTED_LOCATION "${Tbb_${COMPONENT}_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${PC_Tbb_CFLAGS_OTHER}"
        INTERFACE_COMPILE_DEFINITIONS "${Tbb_${COMPONENT}_DEFINITIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${Tbb_INCLUDE_DIR}"
      )
    endif()
  endforeach()
elseif(TBB_FIND_REQUIRED)
  message(FATAL_ERROR "Unable to find TBB")
endif()
