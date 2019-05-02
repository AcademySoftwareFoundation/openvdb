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

Uninstall
---------

Adds a custom target to the CMake build generation which allows for
calling::

  make uninstall

to remove an installation of OpenVDB. Relies on the install_manifest
existing from a previous run of cmake.

#]=======================================================================]

set(MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt")

if(NOT EXISTS ${MANIFEST})
  message(FATAL_ERROR "Cannot find install manifest: '${MANIFEST}'")
endif()

file(STRINGS ${MANIFEST} INSTALLED_FILES)
foreach(INSTALLED_FILE ${INSTALLED_FILES})
  if(EXISTS ${INSTALLED_FILE})
    message(STATUS "Uninstalling: ${INSTALLED_FILE}")
    exec_program(
       ${CMAKE_COMMAND} ARGS "-E remove ${INSTALLED_FILE}"
       OUTPUT_VARIABLE stdout
       RETURN_VALUE RESULT
    )

    if(NOT "${RESULT}" STREQUAL 0)
      message(FATAL_ERROR "Failed to remove file: '${INSTALLED_FILE}'.")
    endif()
  endif()
endforeach(INSTALLED_FILE)
