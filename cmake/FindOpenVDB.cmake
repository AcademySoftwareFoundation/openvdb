# Copyright (c) 2012-2016 DreamWorks Animation LLC
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

# -*- cmake -*-
# - Find OpenVDB
#
# Author : Fredrik Salomonsson fredriks@d2.com
#
# OpenVDB_FOUND                 Set if OpenVDB is found.
# OpenVDB_INCLUDE_DIR           OpenVDB's include directory
# OpenVDB_LIBRARY_DIR           OpenVDB's library directory
# OpenVDB_<C>_LIBRARY           Specific openvdb library (<C> is upper-case)
# OpenVDB_LIBRARIES             All openvdb libraries
# OpenVDB_MAJOR_VERSION         Major version number
# OpenVDB_MINOR_VERSION         Minor version number
# OpenVDB_PATCH_VERSION         Patch version number
#
# This module read hints about search locations from variables::
#
# OPENVDB_ROOT                  Preferred installtion prefix

FIND_PACKAGE( PackageHandleStandardArgs )

FIND_PATH( OPENVDB_LOCATION include/openvdb/version.h 
  "ENV{OPENVDB_ROOT}"
  NO_DEFAULT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  )

FIND_PACKAGE_HANDLE_STANDARD_ARGS( OpenVDB
  REQUIRED_VARS OPENVDB_LOCATION 
  )

IF( OpenVDB_FOUND )
  SET( OpenVDB_INCLUDE_DIR ${OPENVDB_LOCATION}/include
    CACHE PATH "OpenVDB include directory")

  SET( OpenVDB_LIBRARY_DIR ${OPENVDB_LOCATION}/lib
    CACHE PATH "OpenVDB library directory" )
  
  FIND_LIBRARY( OpenVDB_OPENVDB_LIBRARY openvdb
    PATHS ${OpenVDB_LIBRARY_DIR}
    NO_DEFAULT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    )
  
  SET( OpenVDB_LIBRARIES "")
  LIST( APPEND OpenVDB_LIBRARIES ${OpenVDB_OPENVDB_LIBRARY} )
  
  SET( OPENVDB_VERSION_FILE ${OpenVDB_INCLUDE_DIR}/openvdb/version.h )

  FILE( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_major_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+.*")
  FILE( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_minor_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+.*")
  FILE( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_patch_version_str
    REGEX "^#define[\t ]+OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+.*")

  STRING( REGEX REPLACE "^.*OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    _openvdb_major_version_number "${openvdb_major_version_str}")
  STRING( REGEX REPLACE "^.*OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    _openvdb_minor_version_number "${openvdb_minor_version_str}")
  STRING( REGEX REPLACE "^.*OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
    _openvdb_patch_version_number "${openvdb_patch_version_str}")

  SET( OpenVDB_MAJOR_VERSION ${_openvdb_major_version_number}
    CACHE STRING "OpenVDB major version number" )
  SET( OpenVDB_MINOR_VERSION ${_openvdb_minor_version_number}
    CACHE STRING "OpenVDB minor version number" )
  SET( OpenVDB_PATCH_VERSION ${_openvdb_patch_version_number}
    CACHE STRING "OpenVDB patch version number" )

ENDIF( OpenVDB_FOUND )
