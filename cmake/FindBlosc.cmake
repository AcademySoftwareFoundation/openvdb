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

FindBlosc
---------

Find Blosc include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(Blosc
    [version] [EXACT]      # Minimum or EXACT version e.g. 1.5.0
    [REQUIRED]             # Fail with error if Blosc is not found
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``Blosc::blosc``
  This module defines IMPORTED target Blosc::Blosc, if Blosc has been found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Blosc_FOUND``
  True if the system has the Blosc library.
``Blosc_VERSION``
  The version of the Blosc library which was found.
``Blosc_INCLUDE_DIRS``
  Include directories needed to use Blosc.
``Blosc_LIBRARIES``
  Libraries needed to link to Blosc.
``Blosc_LIBRARY_DIRS``
  Blosc library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Blosc_INCLUDE_DIR``
  The directory containing ``blosc.h``.
``Blosc_LIBRARY``
  The path to the Blosc library.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``BLOSC_ROOT``
  Preferred installation prefix.
``BLOSC_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``BLOSC_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

mark_as_advanced(
  Blosc_INCLUDE_DIR
  Blosc_LIBRARY
)

# Append BLOSC_ROOT or $ENV{BLOSC_ROOT} if set (prioritize the direct cmake var)
set(_BLOSC_ROOT_SEARCH_DIR "")

if(BLOSC_ROOT)
  list(APPEND _BLOSC_ROOT_SEARCH_DIR ${BLOSC_ROOT})
else()
  set(_ENV_BLOSC_ROOT $ENV{BLOSC_ROOT})
  if(_ENV_BLOSC_ROOT)
    list(APPEND _BLOSC_ROOT_SEARCH_DIR ${_ENV_BLOSC_ROOT})
  endif()
endif()

# Additionally try and use pkconfig to find blosc

find_package(PkgConfig)
pkg_check_modules(PC_Blosc QUIET blosc)

# ------------------------------------------------------------------------
#  Search for blosc include DIR
# ------------------------------------------------------------------------

set(_BLOSC_INCLUDE_SEARCH_DIRS "")
list(APPEND _BLOSC_INCLUDE_SEARCH_DIRS
  ${BLOSC_INCLUDEDIR}
  ${_BLOSC_ROOT_SEARCH_DIR}
  ${PC_Blosc_INCLUDE_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard blosc header file.
find_path(Blosc_INCLUDE_DIR blosc.h
  NO_DEFAULT_PATH
  PATHS ${_BLOSC_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include
)

if(EXISTS "${Blosc_INCLUDE_DIR}/blosc.h")
  file(STRINGS "${Blosc_INCLUDE_DIR}/blosc.h"
    _blosc_version_major_string REGEX "#define BLOSC_VERSION_MAJOR +[0-9]+ "
  )
  string(REGEX REPLACE "#define BLOSC_VERSION_MAJOR +([0-9]+).*$" "\\1"
    _blosc_version_major_string "${_blosc_version_major_string}"
  )
  string(STRIP "${_blosc_version_major_string}" Blosc_VERSION_MAJOR)

  file(STRINGS "${Blosc_INCLUDE_DIR}/blosc.h"
     _blosc_version_minor_string REGEX "#define BLOSC_VERSION_MINOR +[0-9]+ "
  )
  string(REGEX REPLACE "#define BLOSC_VERSION_MINOR +([0-9]+).*$" "\\1"
    _blosc_version_minor_string "${_blosc_version_minor_string}"
  )
  string(STRIP "${_blosc_version_minor_string}" Blosc_VERSION_MINOR)

  unset(_blosc_version_major_string)
  unset(_blosc_version_minor_string)

  set(Blosc_VERSION ${Blosc_VERSION_MAJOR}.${Blosc_VERSION_MINOR})
endif()

# ------------------------------------------------------------------------
#  Search for blosc lib DIR
# ------------------------------------------------------------------------

set(_BLOSC_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _BLOSC_LIBRARYDIR_SEARCH_DIRS
  ${BLOSC_LIBRARYDIR}
  ${_BLOSC_ROOT_SEARCH_DIR}
  ${PC_Blosc_LIBRARY_DIRS}
  ${SYSTEM_LIBRARY_PATHS}
)

# Static library setup
if(UNIX AND BLOSC_USE_STATIC_LIBS)
  set(_BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

set(BLOSC_PATH_SUFFIXES
  lib64
  lib
)

find_library(Blosc_LIBRARY blosc
  NO_DEFAULT_PATH
  PATHS ${_BLOSC_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${BLOSC_PATH_SUFFIXES}
)

if(UNIX AND BLOSC_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
  unset(_BLOSC_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
endif()

# ------------------------------------------------------------------------
#  Cache and set Blosc_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Blosc
  FOUND_VAR Blosc_FOUND
  REQUIRED_VARS
    Blosc_LIBRARY
    Blosc_INCLUDE_DIR
  VERSION_VAR Blosc_VERSION
)

if(Blosc_FOUND)
  set(Blosc_LIBRARIES ${Blosc_LIBRARY})
  set(Blosc_INCLUDE_DIRS ${Blosc_INCLUDE_DIR})
  set(Blosc_DEFINITIONS ${PC_Blosc_CFLAGS_OTHER})

  get_filename_component(Blosc_LIBRARY_DIRS ${Blosc_LIBRARY} DIRECTORY)

  if(NOT TARGET Blosc::blosc)
    add_library(Blosc::blosc UNKNOWN IMPORTED)
    set_target_properties(Blosc::blosc PROPERTIES
      IMPORTED_LOCATION "${Blosc_LIBRARIES}"
      INTERFACE_COMPILE_DEFINITIONS "${Blosc_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${Blosc_INCLUDE_DIRS}"
    )
  endif()
elseif(Blosc_FIND_REQUIRED)
  message(FATAL_ERROR "Unable to find Blosc")
endif()
