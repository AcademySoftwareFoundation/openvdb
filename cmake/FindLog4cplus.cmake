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

FindLog4cplus
-------------

Find Log4cplus include dirs and libraries

Use this module by invoking find_package with the form::

  find_package(Log4cplus
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if Log4cplus is not found
    )


IMPORTED Targets
^^^^^^^^^^^^^^^^

``Log4cplus::Log4cplus``
  This module defines IMPORTED target Log4cplus::log4cplus, if Log4cplus has been
  found.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Log4cplus_FOUND``
  True if the system has the Log4cplus library.
``Log4cplus_VERSION``
  The version of the Log4cplus library which was found.
``Log4cplus_INCLUDE_DIRS``
  Include directories needed to use Log4cplus.
``Log4cplus_LIBRARIES``
  Libraries needed to link to Log4cplus.
``Log4cplus_LIBRARY_DIRS``
  Log4cplus library directories.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Log4cplus_INCLUDE_DIR``
  The directory containing ``log4cplus/version.h``.
``Log4cplus_LIBRARY``
  The path to the Log4cplus library.

Hints
^^^^^

Instead of explicitly setting the cache variables, the following variables
may be provided to tell this module where to look.

``LOG4CPLUS_ROOT``
  Preferred installation prefix.
``LOG4CPLUS_INCLUDEDIR``
  Preferred include directory e.g. <prefix>/include
``LOG4CPLUS_LIBRARYDIR``
  Preferred library directory e.g. <prefix>/lib
``SYSTEM_LIBRARY_PATHS``
  Paths appended to all include and lib searches.

#]=======================================================================]

mark_as_advanced(
  Log4cplus_INCLUDE_DIR
  Log4cplus_LIBRARY
)

# Append LOG4CPLUS_ROOT or $ENV{LOG4CPLUS_ROOT} if set (prioritize the direct cmake var)
set(_LOG4CPLUS_ROOT_SEARCH_DIR "")

if(LOG4CPLUS_ROOT)
  list(APPEND _LOG4CPLUS_ROOT_SEARCH_DIR ${LOG4CPLUS_ROOT})
else()
  set(_ENV_LOG4CPLUS_ROOT $ENV{LOG4CPLUS_ROOT})
  if(_ENV_LOG4CPLUS_ROOT)
    list(APPEND _LOG4CPLUS_ROOT_SEARCH_DIR ${_ENV_LOG4CPLUS_ROOT})
  endif()
endif()

# Additionally try and use pkconfig to find log4cplus

find_package(PkgConfig)
pkg_check_modules(PC_Log4cplus QUIET log4cplus)

# ------------------------------------------------------------------------
#  Search for Log4cplus include DIR
# ------------------------------------------------------------------------

set(_LOG4CPLUS_INCLUDE_SEARCH_DIRS "")
list(APPEND _LOG4CPLUS_INCLUDE_SEARCH_DIRS
  ${LOG4CPLUS_INCLUDEDIR}
  ${_LOG4CPLUS_ROOT_SEARCH_DIR}
  ${PC_Log4cplus_INCLUDEDIR}
  ${SYSTEM_LIBRARY_PATHS}
)

# Look for a standard log4cplus header file.
find_path(Log4cplus_INCLUDE_DIR log4cplus/version.h
  NO_DEFAULT_PATH
  PATHS ${_LOG4CPLUS_INCLUDE_SEARCH_DIRS}
  PATH_SUFFIXES include
)

if(EXISTS "${Log4cplus_INCLUDE_DIR}/log4cplus/version.h")
  file(STRINGS "${Log4cplus_INCLUDE_DIR}/log4cplus/version.h"
    _log4cplus_version_string REGEX "#define LOG4CPLUS_VERSION LOG4CPLUS_MAKE_VERSION"
  )
  string(REGEX REPLACE "#define LOG4CPLUS_VERSION LOG4CPLUS_MAKE_VERSION\((.*)\).*$" "\\1"
    _log4cplus_version_string "${_log4cplus_version_string}"
  )
  string(REGEX REPLACE "[(]([0-9]+),.*[)].*$" "\\1"
    Log4cplus_MAJOR_VERSION "${_log4cplus_version_string}"
  )
  string(REGEX REPLACE "[(].+, ([0-9]+),.+[)].*$" "\\1"
    Log4cplus_MINOR_VERSION "${_log4cplus_version_string}"
  )
  string(REGEX REPLACE "[(].*,.*, ([0-9]+)[)].*$" "\\1"
    Log4cplus_PATCH_VERSION "${_log4cplus_version_string}"
  )
  unset(_log4cplus_version_string)

  set(Log4cplus_VERSION ${Log4cplus_MAJOR_VERSION}.${Log4cplus_MINOR_VERSION}.${Log4cplus_PATCH_VERSION})
endif()

# ------------------------------------------------------------------------
#  Search for Log4cplus lib DIR
# ------------------------------------------------------------------------

set(_LOG4CPLUS_LIBRARYDIR_SEARCH_DIRS "")
list(APPEND _LOG4CPLUS_LIBRARYDIR_SEARCH_DIRS
  ${LOG4CPLUS_LIBRARYDIR}
  ${_LOG4CPLUS_ROOT_SEARCH_DIR}
  ${PC_Log4cplus_LIBDIR}
  ${SYSTEM_LIBRARY_PATHS}
)


if(UNIX AND LOG4CPLUS_USE_STATIC_LIBS)
  set(_LOG4CPLUS_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

# Build suffix directories

set(LOG4CPLUS_PATH_SUFFIXES
  lib64
  lib
)

find_library(Log4cplus_LIBRARY log4cplus
  NO_DEFAULT_PATH
  PATHS ${_LOG4CPLUS_LIBRARYDIR_SEARCH_DIRS}
  PATH_SUFFIXES ${LOG4CPLUS_PATH_SUFFIXES}
)

if(UNIX AND LOG4CPLUS_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_LOG4CPLUS_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
  unset(_LOG4CPLUS_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
endif()

# ------------------------------------------------------------------------
#  Cache and set Log4cplus_FOUND
# ------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Log4cplus
  FOUND_VAR Log4cplus_FOUND
  REQUIRED_VARS
    Log4cplus_LIBRARY
    Log4cplus_INCLUDE_DIR
  VERSION_VAR Log4cplus_VERSION
)

if(Log4cplus_FOUND)
  set(Log4cplus_LIBRARIES ${Log4cplus_LIBRARY})
  set(Log4cplus_INCLUDE_DIRS ${Log4cplus_INCLUDE_DIR})
  set(Log4cplus_DEFINITIONS ${PC_Log4cplus_CFLAGS_OTHER})

  get_filename_component(Log4cplus_LIBRARY_DIRS ${Log4cplus_LIBRARY} DIRECTORY)

  if(NOT TARGET Log4cplus::log4cplus)
    add_library(Log4cplus::log4cplus UNKNOWN IMPORTED)
    set_target_properties(Log4cplus::log4cplus PROPERTIES
      IMPORTED_LOCATION "${Log4cplus_LIBRARIES}"
      INTERFACE_COMPILE_DEFINITIONS "${Log4cplus_DEFINITIONS}"
      INTERFACE_INCLUDE_DIRECTORIES "${Log4cplus_INCLUDE_DIRS}"
    )
  endif()
elseif(Log4cplus_FIND_REQUIRED)
  message(FATAL_ERROR "Unable to find Log4cplus")
endif()
