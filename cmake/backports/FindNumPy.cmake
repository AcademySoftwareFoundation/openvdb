# Copyright (c) DreamWorks Animation LLC
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

FindNumPy
-----------

Find NumPy include dirs and libraries.

Note that this is limited functionality which has been modified and
back-ported from Kitware's CMake (3.14) python support module which
finds NumPy exclusively by calling methods through a python interpreter.
It's expected that python has already been found through CMake.

Use this module by invoking find_package with the form::

  find_package(NumPy
    [version] [EXACT]      # Minimum or EXACT version
    [REQUIRED]             # Fail with error if NumPy is not found
    )

IMPORTED Targets
^^^^^^^^^^^^^^^^

``Python::NumPy``
  The NumPy library target.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Python_NumPy_FOUND``
  True if the system has the NumPy library.
``Python_NumPy_VERSION``
  The version of the NumPy library which was found.
``Python_NumPy_INCLUDE_DIRS``
  Include directories needed to use NumPy.

#]=======================================================================]

set(_PYTHON_PREFIX Python)

if(NOT ${_PYTHON_PREFIX}_EXECUTABLE)
  message(FATAL_ERROR "Unable to locate NumPy without first locating Python "
    "using find_package(). Alternatively, ensure that the variable "
    "${_PYTHON_PREFIX}_EXECUTABLE is set to your python executable.")
endif()

execute_process(
  COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -c
          "from __future__ import print_function\ntry: import numpy; print(numpy.get_include())\nexcept:pass\n"
  RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
  OUTPUT_VARIABLE _${_PYTHON_PREFIX}_NumPy_PATH
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT _${_PYTHON_PREFIX}_RESULT)
  find_path(${_PYTHON_PREFIX}_NumPy_INCLUDE_DIR
    NAMES "numpy/arrayobject.h" "numpy/numpyconfig.h"
    HINTS "${_${_PYTHON_PREFIX}_NumPy_PATH}"
    NO_DEFAULT_PATH)
endif()
if(${_PYTHON_PREFIX}_NumPy_INCLUDE_DIR)
  set(${_PYTHON_PREFIX}_NumPy_INCLUDE_DIRS "${${_PYTHON_PREFIX}_NumPy_INCLUDE_DIR}")
  set(${_PYTHON_PREFIX}_NumPy_FOUND TRUE)
endif()
if(${_PYTHON_PREFIX}_NumPy_FOUND)
  execute_process(
    COMMAND "${${_PYTHON_PREFIX}_EXECUTABLE}" -c
    "from __future__ import print_function\ntry: import numpy; print(numpy.__version__)\nexcept:pass\n"
    RESULT_VARIABLE _${_PYTHON_PREFIX}_RESULT
    OUTPUT_VARIABLE _${_PYTHON_PREFIX}_NumPy_VERSION)
  if (NOT _${_PYTHON_PREFIX}_RESULT)
     set(${_PYTHON_PREFIX}_NumPy_VERSION "${_${_PYTHON_PREFIX}_NumPy_VERSION}")
  endif()
endif()

# Note no found var here - ${_PYTHON_PREFIX}_NumPy_FOUND is used to match
# later CMake functionality
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy
  REQUIRED_VARS
    ${_PYTHON_PREFIX}_NumPy_INCLUDE_DIR
  VERSION_VAR ${_PYTHON_PREFIX}_NumPy_VERSION
)

if (${_PYTHON_PREFIX}_NumPy_FOUND AND NOT TARGET ${_PYTHON_PREFIX}::NumPy)
  add_library(${_PYTHON_PREFIX}::NumPy INTERFACE IMPORTED)
  set_property(TARGET ${_PYTHON_PREFIX}::NumPy
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${${_PYTHON_PREFIX}_NumPy_INCLUDE_DIR}")
endif()
