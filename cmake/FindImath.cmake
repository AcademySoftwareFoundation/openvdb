# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[.rst:

FindImath
-----------

Find Imath include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(Imath
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if Imath is not found
    [COMPONENTS <libs>...] # Imath libraries by their canonical name
                           # e.g. "Half" for "libHalf"
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``Imath::Imath``
  The Imath library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Imath_FOUND``
  True if the system has the Imath library.
``Imath_VERSION``
  The version of the Imath library which was found.
``Imath_INCLUDE_DIRS``
  Include directories needed to use Imath.
``Imath_RELEASE_LIBRARIES``
  Libraries needed to link to the release version of Imath.
``Imath_RELEASE_LIBRARY_DIRS``
  Imath release library directories.
``Imath_DEBUG_LIBRARIES``
  Libraries needed to link to the debug version of Imath.
``Imath_DEBUG_LIBRARY_DIRS``
  Imath debug library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Imath_INCLUDE_DIR``
  The directory containing ``Imath/config-auto.h``.
``Imath_LIBRARY``
  Individual component libraries for Imath. may include target_link_libraries() debug/optimized keywords.
``Imath_LIBRARY_RELEASE``
  Individual component libraries for Imath release
``Imath_LIBRARY_DEBUG``
  Individual component libraries for Imath debug

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``Imath_ROOT``
  Preferred installation prefix.
``IMATH_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``IMATH_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``IMATH_DEBUG_SUFFIX``
  Suffix of the debug version of ilmbase libs. Defaults to "_d".
``SYSTEM_LIBRARY_PATHS``
  Global list of library paths intended to be searched by and find_xxx call
``IMATH_USE_STATIC_LIBS``
  Only search for static ilmbase libraries
``DISABLE_CMAKE_SEARCH_PATHS``
  Disable CMakes default search paths for find_xxx calls in this module

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
include(GNUInstallDirs)

set(IMATH_REPO "https://github.com/AcademySoftwareFoundation/Imath.git" CACHE STRING
    "Repo for auto-build of Imath")
set(IMATH_TAG "master" CACHE STRING
  "Tag for auto-build of Imath (branch, tag, or SHA)")

find_package(Imath CONFIG QUIET)

if(NOT TARGET Imath::Imath AND NOT Imath_FOUND)

  include(FetchContent)
  FetchContent_Declare(Imath
    GIT_REPOSITORY ${IMATH_REPO}
    GIT_TAG ${IMATH_TAG}
    GIT_SHALLOW ON
      )
    
  FetchContent_GetProperties(Imath)
  if(NOT Imath_POPULATED)
    FetchContent_Populate(Imath)
    # hrm, cmake makes Imath lowercase for the properties (to imath)
    add_subdirectory(${imath_SOURCE_DIR} ${imath_BINARY_DIR})
  endif()
  # the install creates this but if we're using the library locally we
  # haven't installed the header files yet, so need to extract those
  # and make a variable for header only usage
  if(NOT TARGET Imath::ImathConfig)
    get_target_property(imathinc Imath INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(imathconfinc ImathConfig INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND imathinc ${imathconfinc})
    set(IMATH_HEADER_ONLY_INCLUDE_DIRS ${imathinc})
    message(STATUS "Imath interface dirs ${IMATH_HEADER_ONLY_INCLUDE_DIRS}")
  endif()
else()
  message(STATUS "Using Imath from ${Imath_DIR}")
endif()

