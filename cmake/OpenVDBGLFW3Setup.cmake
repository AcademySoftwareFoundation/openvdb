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

OpenVDBGLFW3Setup
---------

Wraps the call the FindPackage ( glfw3 ) for OpenVDB builds. Provides
some extra options for finding the glfw3 installation without polluting
the OpenVDB Binaries cmake.

Use this module by invoking include with the form::

  include ( OpenVDBGLFW3Setup )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``glfw``
  The glfw3 library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``glfw3_FOUND``
  True if the system has glfw3 installed.
``glfw3_VERSION``
  The version of the glfw3 library which was found.

Hints
^^^^^^^^^^^^^^^

The following variables may be provided to tell this module where to look.

``GLFW3_ROOT``
  Preferred installation prefix.

#]=======================================================================]

# Find the glfw3 installation and use glfw's CMake to initialize
# the glfw lib

SET ( _GLFW3_ROOT_SEARCH_DIR "" )

IF ( GLFW3_ROOT )
  LIST ( APPEND _GLFW3_ROOT_SEARCH_DIR ${GLFW3_ROOT} )
ELSE ()
  SET ( _ENV_GLFW_ROOT $ENV{GLFW3_ROOT} )
  IF ( _ENV_GLFW_ROOT )
    LIST ( APPEND _GLFW3_ROOT_SEARCH_DIR ${_ENV_GLFW_ROOT} )
  ENDIF ()
ENDIF ()

# Additionally try and use pkconfig to find glfw, though we only use
# pkg-config to re-direct to the cmake. In other words, glfw's cmake is
# expected to be installed
FIND_PACKAGE ( PkgConfig )
PKG_CHECK_MODULES ( PC_glfw3 QUIET glfw3 )

IF ( PC_glfw3_FOUND )
  FOREACH ( DIR ${PC_glfw3_LIBRARY_DIRS} )
    LIST ( APPEND _GLFW3_ROOT_SEARCH_DIR ${DIR} )
  ENDFOREACH ()
ENDIF ()

FIND_PATH ( GLFW3_CMAKE_LOCATION glfw3Config.cmake
  NO_DEFAULT_PATH
  PATHS ${_GLFW3_ROOT_SEARCH_DIR}
  PATH_SUFFIXES lib/cmake/glfw3 cmake/glfw3 glfw3
  )

IF ( GLFW3_CMAKE_LOCATION )
  LIST ( APPEND CMAKE_PREFIX_PATH "${GLFW3_CMAKE_LOCATION}" )
ENDIF ()

FIND_PACKAGE ( glfw3 ${MINIMUM_GLFW_VERSION} REQUIRED )

FIND_PACKAGE ( PackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS ( glfw3
  REQUIRED_VARS glfw3_DIR glfw3_FOUND
  VERSION_VAR glfw3_VERSION
  )
