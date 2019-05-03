# Copyright (c) 2012-2019 DreamWorks Animation LLC
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DreamWorks Animation nor the names of
# its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
# LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
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
``IlmBase_LIBRARIES``
  Libraries needed to link to IlmBase.
``IlmBase_LIBRARY_DIRS``
  IlmBase library directories.
``IlmBase_{COMPONENT}_FOUND``
  True if the system has the named IlmBase component.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``IlmBase_INCLUDE_DIR``
  The directory containing ``IlmBase/config-auto.h``.
``IlmBase_{COMPONENT}_LIBRARY``
  Individual component libraries for IlmBase
``IlmBase_{COMPONENT}_DLL``
  Individual component dlls for IlmBase on Windows.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``ILMBASE_ROOT``
  Preferred installation prefix.
``ILMBASE_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``ILMBASE_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

# Support new if() IN_LIST operator
if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

mark_as_advanced(
  IlmBase_INCLUDE_DIR
  IlmBase_LIBRARY
)

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

# Append ILMBASE_ROOT or $ENV{ILMBASE_ROOT} if set (prioritize the direct cmake var)
set(_ILMBASE_ROOT_SEARCH_DIR "")

if(ILMBASE_ROOT)
  list(APPEND _ILMBASE_ROOT_SEARCH_DIR ${ILMBASE_ROOT})
else()
  set(_ENV_ILMBASE_ROOT $ENV{ILMBASE_ROOT})
  if(_ENV_ILMBASE_ROOT)
    list(APPEND _ILMBASE_ROOT_SEARCH_DIR ${_ENV_ILMBASE_ROOT})
  endif()
endif()

# Additionally try and use pkconfig to find IlmBase

find_package(PkgConfig)
pkg_check_modules(PC_IlmBase QUIET IlmBase)

# ------------------------------------------------------------------------
#  Search for IlmBase include DIR
# ------------------------------------------------------------------------

set(_ILMBASE_INCLUDE_SEARCH_DIRS "")
list(APPEND _ILMBASE_INCLUDE_SEARCH_DIRS
  ${ILMBASE_INCLUDEDIR}
  ${_ILMBASE_ROOT_SEARCH_DIR}
  ${PC_IlmBase_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard IlmBase header file.
find_path(IlmBase_INCLUDE_DIR IlmBaseConfig.h
  NO_DEFAULT_PATH
  PATHS ${_ILMBASE_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include/OpenEXR OpenEXR
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

set(_ILMBASE_LIBRARYDIR_SEARCH_DIRS "")

# Append to _ILMBASE_LIBRARYDIR_SEARCH_DIRS in priority order

list(APPEND _ILMBASE_LIBRARYDIR_SEARCH_DIRS
  ${ILMBASE_LIBRARYDIR}
  ${_ILMBASE_ROOT_SEARCH_DIR}
  ${PC_IlmBase_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Build suffix directories

set(ILMBASE_PATH_SUFFIXES
  lib64
  lib
)

if(UNIX)
  list(INSERT ILMBASE_PATH_SUFFIXES 0 lib/x86_64-linux-gnu)
endif()

set(_ILMBASE_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

# library suffix handling
if(WIN32)
  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES
    "-${IlmBase_VERSION_MAJOR}_${IlmBase_VERSION_MINOR}.lib"
  )
else()
  if(ILMBASE_USE_STATIC_LIBS)
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES
      "-${IlmBase_VERSION_MAJOR}_${IlmBase_VERSION_MINOR}.a"
    )
  else()
    if(APPLE)
      list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES
        "-${IlmBase_VERSION_MAJOR}_${IlmBase_VERSION_MINOR}.dylib"
      )
    else()
      list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES
        "-${IlmBase_VERSION_MAJOR}_${IlmBase_VERSION_MINOR}.so"
      )
    endif()
  endif()
endif()

set(IlmBase_LIB_COMPONENTS "")

foreach(COMPONENT ${IlmBase_FIND_COMPONENTS})
  find_library(IlmBase_${COMPONENT}_LIBRARY ${COMPONENT}
    NO_DEFAULT_PATH
    PATHS ${_ILMBASE_LIBRARYDIR_SEARCH_DIRS}
    PATH_SUFFIXES ${ILMBASE_PATH_SUFFIXES}
  )
  list(APPEND IlmBase_LIB_COMPONENTS ${IlmBase_${COMPONENT}_LIBRARY})

  if(WIN32 AND NOT ILMBASE_USE_STATIC_LIBS)
    set(_ILMBASE_TMP ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll")
    find_library(IlmBase_${COMPONENT}_DLL ${COMPONENT}
      NO_DEFAULT_PATH
      PATHS ${_ILMBASE_LIBRARYDIR_SEARCH_DIRS}
      PATH_SUFFIXES bin
    )
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ILMBASE_TMP})
    unset(_ILMBASE_TMP)
  endif()

  if(IlmBase_${COMPONENT}_LIBRARY)
    set(IlmBase_${COMPONENT}_FOUND TRUE)
  else()
    set(IlmBase_${COMPONENT}_FOUND FALSE)
  endif()
endforeach()

# reset lib suffix

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ILMBASE_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
unset(_ILMBASE_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

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

if(IlmBase_FOUND)
  set(IlmBase_LIBRARIES ${IlmBase_LIB_COMPONENTS})

  # We have to add both include and include/OpenEXR to the include
  # path in case OpenEXR and IlmBase are installed separately

  set(IlmBase_INCLUDE_DIRS)
  list(APPEND IlmBase_INCLUDE_DIRS
    ${IlmBase_INCLUDE_DIR}/../
    ${IlmBase_INCLUDE_DIR}
  )
  set(IlmBase_DEFINITIONS ${PC_IlmBase_CFLAGS_OTHER})

  set(IlmBase_LIBRARY_DIRS "")
  foreach(LIB ${IlmBase_LIB_COMPONENTS})
    get_filename_component(_ILMBASE_LIBDIR ${LIB} DIRECTORY)
    list(APPEND IlmBase_LIBRARY_DIRS ${_ILMBASE_LIBDIR})
  endforeach()
  list(REMOVE_DUPLICATES IlmBase_LIBRARY_DIRS)

  # Configure imported targets

  foreach(COMPONENT ${IlmBase_FIND_COMPONENTS})
    if(NOT TARGET IlmBase::${COMPONENT})
      add_library(IlmBase::${COMPONENT} UNKNOWN IMPORTED)
      set_target_properties(IlmBase::${COMPONENT} PROPERTIES
        IMPORTED_LOCATION "${IlmBase_${COMPONENT}_LIBRARY}"
        INTERFACE_COMPILE_OPTIONS "${IlmBase_DEFINITIONS}"
        INTERFACE_INCLUDE_DIRECTORIES "${IlmBase_INCLUDE_DIRS}"
      )
    endif()
  endforeach()

elseif(IlmBase_FIND_REQUIRED)
  message(FATAL_ERROR "Unable to find IlmBase")
endif()
