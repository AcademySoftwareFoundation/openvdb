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
#[=======================================================================[

  CMake Configuration for OpenVDB Maya

#]=======================================================================]

PROJECT ( OpenVDBMaya )
CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )
# Monitoring <PackageName>_ROOT variables
IF ( POLICY CMP0074 )
  CMAKE_POLICY ( SET CMP0074 NEW )
ENDIF ()

###### OpenVDB Maya Options

OPTION ( OPENVDB_MAYA_INSTALL_MOD "Install module file" ON )

IF ( OPENVDB_MAYA_SUBDIR )
  MESSAGE ( DEPRECATION "OPENVDB_MAYA_SUBDIR is deprecated and will be removed. "
    "Set OPENVDB_MAYA_INSTALL_PREFIX to your install location."
    )
ENDIF ()

##########################################################################

MESSAGE ( STATUS "----------------------------------------------------" )
MESSAGE ( STATUS "------------- Configuring OpenVDBMaya --------------" )
MESSAGE ( STATUS "----------------------------------------------------" )

#########################################################################

# Configure prefix

IF ( NOT OPENVDB_MAYA_INSTALL_PREFIX )
  IF ( OPENVDB_MAYA_SUBDIR )
    SET ( MAYA_INSTALL_LOCATION
      ${CMAKE_INSTALL_PREFIX}/maya/${Maya_VERSION}
      )
  ELSE ()
    SET ( MAYA_INSTALL_LOCATION
      ${CMAKE_INSTALL_PREFIX}/maya${Maya_VERSION}
      )
  ENDIF ()
ENDIF ()

SET ( OPENVDB_MAYA_INSTALL_PREFIX ${MAYA_INSTALL_LOCATION}
  CACHE PATH "Base install path for OpenVDB Maya libraries and headers."
  )

MESSAGE ( STATUS "Maya base install path: ${OPENVDB_MAYA_INSTALL_PREFIX}" )

#########################################################################

# Collect and configure lib dependencies

IF ( NOT OPENVDB_BUILD_CORE )
  SET ( OPENVDB_LIB OpenVDB::openvdb )
ELSE ()
  SET ( OPENVDB_LIB openvdb_shared )
ENDIF ()

FIND_PACKAGE ( OpenGL REQUIRED )

SET ( OPENVDB_MAYA_DEPENDENT_LIBS
  ${OPENVDB_LIB}
  Maya
  OpenGL::GL
)

#########################################################################

SET ( OPENVDB_MAYA_MODULE_SOURCE_FILES
  maya/OpenVDBCopyNode.cc
  maya/OpenVDBData.cc
  maya/OpenVDBFilterNode.cc
  maya/OpenVDBFromMayaFluidNode.cc
  maya/OpenVDBFromPolygonsNode.cc
  maya/OpenVDBPlugin.cc
  maya/OpenVDBReadNode.cc
  maya/OpenVDBToPolygonsNode.cc
  maya/OpenVDBTransformNode.cc
  maya/OpenVDBUtil.cc
  maya/OpenVDBVisualizeNode.cc
  maya/OpenVDBWriteNode.cc
  )

SET ( OPENVDB_MAYA_MODULE_MEL_FILES
  maya/AEOpenVDBCopyTemplate.mel
  maya/AEOpenVDBFilterTemplate.mel
  maya/AEOpenVDBFromMayaFluidTemplate.mel
  maya/AEOpenVDBFromPolygonsTemplate.mel
  maya/AEOpenVDBReadTemplate.mel
  maya/AEOpenVDBToPolygonsTemplate.mel
  maya/AEOpenVDBTransformTemplate.mel
  maya/AEOpenVDBVisualizeTemplate.mel
  maya/AEOpenVDBWriteTemplate.mel
 )

#########################################################################

# This is to work around DWA way of doing stuff
SET ( OPENVDB_MAYA_HEADERS
  "maya/OpenVDBData.h"
  "maya/OpenVDBUtil.h"
  )

SET ( OPENVDB_MAYA_DIR ${PROJECT_BINARY_DIR}/openvdb_maya )
FILE ( MAKE_DIRECTORY ${OPENVDB_MAYA_DIR} )
FILE ( COPY ${OPENVDB_MAYA_HEADERS} DESTINATION ${OPENVDB_MAYA_DIR} )

#########################################################################

ADD_LIBRARY ( openvdb_maya SHARED
  ${OPENVDB_MAYA_MODULE_SOURCE_FILES}
  )

TARGET_INCLUDE_DIRECTORIES ( openvdb_maya PUBLIC ${PROJECT_BINARY_DIR} )
TARGET_LINK_LIBRARIES ( openvdb_maya ${OPENVDB_MAYA_DEPENDENT_LIBS} )

MAYA_SET_LIBRARY_PROPERTIES ( openvdb_maya )

SET_TARGET_PROPERTIES ( openvdb_maya
  PROPERTIES
  OUTPUT_NAME OpenVDB
  )

IF ( OPENVDB_ENABLE_RPATH )
  SET ( RPATHS "" )
  IF ( OPENVDB_BUILD_CORE )
    LIST ( APPEND RPATHS ${CMAKE_INSTALL_PREFIX}/lib )
  ELSE ()
    LIST ( APPEND RPATHS ${OpenVDB_LIBRARY_DIRS} )
  ENDIF ()

  LIST ( REMOVE_DUPLICATES RPATHS )
  SET_TARGET_PROPERTIES ( openvdb_maya
    PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  UNSET ( RPATHS )
ENDIF ()

#########################################################################

INSTALL ( TARGETS
  openvdb_maya
  DESTINATION
  ${OPENVDB_MAYA_INSTALL_PREFIX}/plug-ins
  )

INSTALL ( FILES
  ${OPENVDB_MAYA_MODULE_MEL_FILES}
  DESTINATION
  ${OPENVDB_MAYA_INSTALL_PREFIX}/scripts
  )

IF ( OPENVDB_MAYA_INSTALL_MOD )
  INSTALL ( CODE
    "FILE (WRITE ${OPENVDB_MAYA_INSTALL_PREFIX}/openvdb.mod \"+ openvdb any ${OPENVDB_MAYA_INSTALL_PREFIX}\n\")"
    COMPONENT Runtime
    )
ENDIF ()
