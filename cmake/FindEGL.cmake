# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0

# Try to find EGL library and include dir.
# Once done this will define
#
# EGL_FOUND        - true if EGL has been found
# EGL_INCLUDE_DIR  - where the EGL/egl.h and KHR/khrplatform.h can be found
# EGL_LIBRARY      - link this to use libEGL.so.1
# EGL_opengl_LIBRARY     - link with these two libraries instead of the gl library
# EGL_gldispatch_LIBRARY   for full OpenGL support through EGL
# EGL_LIBRARIES    - all EGL related libraries: EGL, OpenGL, GLdispatch


if(NOT EGL_INCLUDE_DIR)

  # If we have a root defined look there first
  if(EGL_ROOT)
    find_path(EGL_INCLUDE_DIR EGL/egl.h PATHS ${EGL_ROOT}/include
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT EGL_INCLUDE_DIR)
    find_path(EGL_INCLUDE_DIR EGL/egl.h PATHS
      /usr/local/include
      /usr/include
    )
  endif()
endif()

if(NOT EGL_LIBRARY)
  # If we have a root defined look there first
  if(EGL_ROOT)
    find_library(EGL_LIBRARY EGL PATHS ${EGL_ROOT}/lib
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT EGL_LIBRARY)
    find_library(EGL_LIBRARY EGL PATHS
      /usr/local/lib
      /usr/lib
    )
  endif()
endif()

if(NOT EGL_opengl_LIBRARY)
  # If we have a root defined look there first
  if(EGL_ROOT)
    find_library(EGL_opengl_LIBRARY OpenGL PATHS ${EGL_ROOT}/lib
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT EGL_opengl_LIBRARY)
    find_library(EGL_opengl_LIBRARY OpenGL PATHS
      /usr/local/lib
      /usr/lib
    )
  endif()
endif()

if(NOT EGL_gldispatch_LIBRARY)
  # If we have a root defined look there first
  if(EGL_ROOT)
    find_library(EGL_gldispatch_LIBRARY GLdispatch PATHS ${EGL_ROOT}/lib
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT EGL_gldispatch_LIBRARY)
    find_library(EGL_gldispatch_LIBRARY GLdispatch PATHS
      /usr/local/lib
      /usr/lib
    )
  endif()
endif()

set(EGL_LIBRARIES ${EGL_LIBRARY} ${EGL_opengl_LIBRARY} ${EGL_gldispatch_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EGL  DEFAULT_MSG
                                  EGL_LIBRARY  EGL_opengl_LIBRARY EGL_gldispatch_LIBRARY EGL_INCLUDE_DIR)

mark_as_advanced(EGL_DIR EGL_INCLUDE_DIR EGL_LIBRARY EGL_opengl_LIBRARY EGL_gldispatch_LIBRARY)
