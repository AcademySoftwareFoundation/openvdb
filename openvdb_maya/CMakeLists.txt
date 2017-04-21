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
PROJECT ( OpenVDBMaya )

OPTION ( OPENVDB_MAYA_SUBDIR "Install plug-ins to <prefix>/maya/<version> instead of <prefix>/maya<version>" OFF)
OPTION ( OPENVDB_MAYA_INSTALL_MOD "Install module file" ON)

FIND_PACKAGE ( Maya REQUIRED )
FIND_PACKAGE ( Boost ${MINIMUM_BOOST_VERSION} REQUIRED )
FIND_PACKAGE ( ILMBase REQUIRED )

IF ( NOT OPENVDB_BUILD_CORE )
  FIND_PACKAGE ( OpenVDB REQUIRED )

  # Use the library installed
  SET ( OPENVDB_SHARED_LIB ${OpenVDB_OPENVDB_LIBRARY} )
ELSE ()
  # Use the target
  SET ( OPENVDB_SHARED_LIB openvdb_shared)
ENDIF ()

# Include them as system headers to avoid getting spammed by internal
# warnings
INCLUDE_DIRECTORIES ( SYSTEM ${MAYA_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( SYSTEM ${Boost_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( SYSTEM ${ILMBASE_INCLUDE_DIRS} )

ADD_DEFINITIONS ( ${MAYA_DEFINITIONS} )

# This is to work around DWA way of doing stuff
LIST ( APPEND OPENVDB_MAYA_HEADER "maya/OpenVDBData.h" )
LIST ( APPEND OPENVDB_MAYA_HEADER "maya/OpenVDBUtil.h" )
SET  ( OPENVDB_MAYA_DIR ${PROJECT_BINARY_DIR}/openvdb_maya )
FILE ( MAKE_DIRECTORY ${OPENVDB_MAYA_DIR} )
FILE ( COPY ${OPENVDB_MAYA_HEADER} DESTINATION ${OPENVDB_MAYA_DIR} )
INCLUDE_DIRECTORIES ( ${PROJECT_BINARY_DIR} )

# RPath handling
IF ( OPENVDB_ENABLE_RPATH )

  # when building, don't use the install RPATH already
  # (but later on when installing)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

  # Use $ORIGIN for relative referencing
  SET(CMAKE_INSTALL_RPATH "$ORIGIN/../../lib")

ENDIF ( OPENVDB_ENABLE_RPATH )

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

SET_SOURCE_FILES_PROPERTIES ( ${OPENVDB_MAYA_MODULE_SOURCE_FILES}
  PROPERTIES
  COMPILE_FLAGS "-DOPENVDB_PRIVATE -DOPENVDB_USE_BLOSC"
  )

ADD_LIBRARY ( openvdb_maya SHARED
  ${OPENVDB_MAYA_MODULE_SOURCE_FILES}
  )

TARGET_LINK_LIBRARIES ( openvdb_maya
  ${OPENVDB_SHARED_LIB}
  ${MAYA_OpenMayaFX_LIBRARY}
  ${MAYA_OpenMayaUI_LIBRARY}
  ${MAYA_OpenMaya_LIBRARY}
  ${MAYA_Foundation_LIBRARY}
  )

SET_TARGET_PROPERTIES ( openvdb_maya
  PROPERTIES
  OUTPUT_NAME OpenVDB
  )

MAYA_SET_LIBRARY_PROPERTIES ( openvdb_maya )

SET( OPENVDB_MAYA_INSTALL_BASE_DIR
  maya$<$<BOOL:${OPENVDB_MAYA_SUBDIR}>:/>${MAYA_VERSION_NUMBER}
  CACHE PATH "Openvdb's maya install directory")

INSTALL ( TARGETS
  openvdb_maya
  DESTINATION
  ${OPENVDB_MAYA_INSTALL_BASE_DIR}/plug-ins
  )

INSTALL ( FILES
  maya/AEOpenVDBCopyTemplate.mel
  maya/AEOpenVDBFilterTemplate.mel
  maya/AEOpenVDBFromMayaFluidTemplate.mel
  maya/AEOpenVDBFromPolygonsTemplate.mel
  maya/AEOpenVDBReadTemplate.mel
  maya/AEOpenVDBToPolygonsTemplate.mel
  maya/AEOpenVDBTransformTemplate.mel
  maya/AEOpenVDBVisualizeTemplate.mel
  maya/AEOpenVDBWriteTemplate.mel
  DESTINATION
  ${OPENVDB_MAYA_INSTALL_BASE_DIR}/scripts
  )

IF ( OPENVDB_MAYA_INSTALL_MOD )
  # Cmake doesn't evaluate the generator-expression.
  IF ( OPENVDB_MAYA_SUBDIR )
    INSTALL ( CODE
      "FILE (WRITE ${CMAKE_INSTALL_PREFIX}/maya/${MAYA_VERSION_NUMBER}/openvdb.mod \"+ openvdb any ${CMAKE_INSTALL_PREFIX}/maya/${MAYA_VERSION_NUMBER}\n\")
  "
      COMPONENT Runtime
      )
  ELSE ()
    INSTALL ( CODE
      "FILE (WRITE ${CMAKE_INSTALL_PREFIX}/maya${MAYA_VERSION_NUMBER}/openvdb.mod \"+ openvdb any ${CMAKE_INSTALL_PREFIX}/maya${MAYA_VERSION_NUMBER}\n\")
  "
      COMPONENT Runtime
      )
  ENDIF ( OPENVDB_MAYA_SUBDIR )
ENDIF ( OPENVDB_MAYA_INSTALL_MOD )
