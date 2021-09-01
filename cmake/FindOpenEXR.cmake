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
                           # e.g. "OpenEXR" for "libOpenEXR"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``OpenEXR::OpenEXR``
  The OpenEXR library target.
``OpenEXR::OpenEXRUtil``
  The OpenEXRUtil library target.

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

cmake_minimum_required(VERSION 3.12)
include(GNUInstallDirs)

set(OPENEXR_REPO "https://github.com/AcademySoftwareFoundation/openexr.git" CACHE STRING
    "Repo for auto-build of OpenEXR")
set(OPENEXR_TAG "master" CACHE STRING
  "Tag for auto-build of OpenEXR (branch, tag, or SHA)")

find_package(OpenEXR CONFIG QUIET)

if(NOT TARGET OpenEXR::OpenEXR AND NOT OpenEXR_FOUND)

  include(FetchContent)
  FetchContent_Declare(OpenEXR
    GIT_REPOSITORY ${OPENEXR_REPO}
    GIT_TAG ${OPENEXR_TAG}
    GIT_SHALLOW ON
      )
    
  FetchContent_GetProperties(OpenEXR)
  if(NOT OpenEXR_POPULATED)
    FetchContent_Populate(OpenEXR)
    # hrm, cmake makes OpenEXR lowercase for the properties (to openexr)
    add_subdirectory(${openexr_SOURCE_DIR} ${openexr_BINARY_DIR})
  endif()
  # the install creates this but if we're using the library locally we
  # haven't installed the header files yet, so need to extract those
  # and make a variable for header only usage
  if(NOT TARGET OpenEXR::OpenEXRConfig)
    get_target_property(openexrinc OpenEXR INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(openexrconfinc OpenEXRConfig INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND openexrinc ${openexrconfinc})
    set(OPENEXR_HEADER_ONLY_INCLUDE_DIRS ${openexrinc})
    message(STATUS "OpenEXR interface dirs ${OPENEXR_HEADER_ONLY_INCLUDE_DIRS}")
  endif()
else()
  message(STATUS "Using OpenEXR from ${OpenEXR_DIR}")
endif()

