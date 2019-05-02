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

OpenVDBUtils.cmake
------------------

A utility CMake file which provides helper functions for configuring an
OpenVDB installation.

Use this module by invoking include with the form::

  include ( OpenVDBUtils )


The following functions are provided:

``OPENVDB_VERSION_FROM_HEADER``

  OPENVDB_VERSION_FROM_HEADER ( <header_path>
    VERSION [<version>]
    MAJOR   [<version>]
    MINOR   [<version>]
    PATCH   [<version>] )

  Parse the provided version file to retrieve the current OpenVDB
  version information. The file is expected to be a version.h file
  as found in the following path of an OpenVDB repository:
    openvdb/version.h

  If the file does not exist, variables are unmodified.

``OPENVDB_ABI_VERSION_FROM_PRINT``

  OPENVDB_ABI_VERSION_FROM_PRINT ( <vdb_print>
    [QUIET]
    ABI [<version>] )

  Retrieve the ABI version that an installation of OpenVDB was compiled
  for using the provided vdb_print binary. Parses the result of:
    vdb_print --version

  If the binary does not exist or fails to launch, variables are
  unmodified.

#]=======================================================================]


FUNCTION ( OPENVDB_VERSION_FROM_HEADER OPENVDB_VERSION_FILE )
  CMAKE_PARSE_ARGUMENTS ( _VDB "" "VERSION;MAJOR;MINOR;PATCH" "" ${ARGN} )

  IF ( NOT EXISTS ${OPENVDB_VERSION_FILE} )
    RETURN ()
  ENDIF ()

  FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+.*"
    )
  STRING ( REGEX REPLACE "^.*OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    _OpenVDB_MAJOR_VERSION "${openvdb_version_str}"
    )

  FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+.*"
    )
  STRING ( REGEX REPLACE "^.*OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    _OpenVDB_MINOR_VERSION "${openvdb_version_str}"
    )

  FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+.*"
    )
  STRING ( REGEX REPLACE "^.*OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    _OpenVDB_PATCH_VERSION "${openvdb_version_str}"
    )
  UNSET ( openvdb_version_str )

  IF ( _VDB_VERSION )
    SET ( ${_VDB_VERSION}
      ${_OpenVDB_MAJOR_VERSION}.${_OpenVDB_MINOR_VERSION}.${_OpenVDB_PATCH_VERSION}
      PARENT_SCOPE
      )
  ENDIF  ()
  IF ( _VDB_MAJOR )
    SET ( ${_VDB_MAJOR} ${_OpenVDB_MAJOR_VERSION} PARENT_SCOPE )
  ENDIF ()
  IF ( _VDB_MINOR )
    SET ( ${_VDB_MINOR} ${_OpenVDB_MINOR_VERSION} PARENT_SCOPE )
  ENDIF ()
  IF ( _VDB_PATCH )
    SET ( ${_VDB_PATCH} ${_OpenVDB_PATCH_VERSION} PARENT_SCOPE )
  ENDIF ()
ENDFUNCTION ()


########################################################################
########################################################################


FUNCTION ( OPENVDB_ABI_VERSION_FROM_PRINT OPENVDB_PRINT )
  CMAKE_PARSE_ARGUMENTS ( _VDB "QUIET" "ABI" "" ${ARGN} )

  IF ( NOT EXISTS ${OPENVDB_PRINT} )
    RETURN ()
  ENDIF ()

  SET ( _VDB_PRINT_VERSION_STRING "" )
  SET ( _VDB_PRINT_RETURN_STATUS "" )

  IF ( ${_VDB_QUIET} )
    EXECUTE_PROCESS ( COMMAND ${OPENVDB_PRINT} "--version"
      RESULT_VARIABLE _VDB_PRINT_RETURN_STATUS
      OUTPUT_VARIABLE _VDB_PRINT_VERSION_STRING
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
  ELSE ()
    EXECUTE_PROCESS ( COMMAND ${OPENVDB_PRINT} "--version"
      RESULT_VARIABLE _VDB_PRINT_RETURN_STATUS
      OUTPUT_VARIABLE _VDB_PRINT_VERSION_STRING
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
  ENDIF ()

  IF ( ${_VDB_PRINT_RETURN_STATUS} )
    RETURN ()
  ENDIF ()

  SET ( _OpenVDB_ABI )
  STRING ( REGEX REPLACE ".*abi([0-9]*).*" "\\1" _OpenVDB_ABI ${_VDB_PRINT_VERSION_STRING} )
  UNSET ( _VDB_PRINT_RETURN_STATUS )
  UNSET ( _VDB_PRINT_VERSION_STRING )

  IF ( _VDB_ABI )
    SET ( ${_VDB_ABI} ${_OpenVDB_ABI} PARENT_SCOPE )
  ENDIF ()
ENDFUNCTION ()
