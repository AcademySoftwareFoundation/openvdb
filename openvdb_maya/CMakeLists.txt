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

OPTION ( OPENVDB_MAYA_SUBDIR "Install plug-ins to <prefix>/maya/<version> instead of <prefix>/maya<version>" OFF )
OPTION ( OPENVDB_MAYA_INSTALL_MOD "Install module file" ON )

##########################################################################

MESSAGE ( STATUS "----------------------------------------------------" )
MESSAGE ( STATUS "------------- Configuring OpenVDBMaya- -------------" )
MESSAGE ( STATUS "----------------------------------------------------" )

##########################################################################

# Collect and configure lib dependencies

IF ( NOT OPENVDB_BUILD_CORE )
  FIND_PACKAGE ( OpenVDB REQUIRED )
  SET ( OPENVDB_SHARED_LIB OpenVDB::openvdb )
ELSE ()
  SET ( OPENVDB_SHARED_LIB openvdb_shared )
  INCLUDE_DIRECTORIES ( ../ )
ENDIF ()

FIND_PACKAGE ( Maya REQUIRED )
FIND_PACKAGE ( Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS iostreams system )
FIND_PACKAGE ( IlmBase REQUIRED COMPONENTS Half )

SET ( OPENVDB_MAYA_DEPENDANT_LIBS
  ${OPENVDB_SHARED_LIB}
  Boost::iostreams
  Boost::system
  IlmBase::Half
)

IF ( USE_BLOSC )
  FIND_PACKAGE ( Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED )
  LIST ( APPEND OPENVDB_MAYA_DEPENDANT_LIBS Blosc::blosc )
ENDIF ()

# @todo sort out tbb for maya - it's provided with maya and we should be using
# those includes/libs. This will conflict if attempting to build the maya and
# Houdini plugin

INCLUDE_DIRECTORIES ( SYSTEM ${MAYA_INCLUDE_DIR} )
ADD_DEFINITIONS ( ${MAYA_DEFINITIONS} )

# This is to work around DWA way of doing stuff
LIST ( APPEND OPENVDB_MAYA_HEADER "maya/OpenVDBData.h" )
LIST ( APPEND OPENVDB_MAYA_HEADER "maya/OpenVDBUtil.h" )
SET  ( OPENVDB_MAYA_DIR ${PROJECT_BINARY_DIR}/openvdb_maya )
FILE ( MAKE_DIRECTORY ${OPENVDB_MAYA_DIR} )
FILE ( COPY ${OPENVDB_MAYA_HEADER} DESTINATION ${OPENVDB_MAYA_DIR} )

INCLUDE_DIRECTORIES ( ${PROJECT_BINARY_DIR} )

##########################################################################

SET( OPENVDB_MAYA_INSTALL_BASE_DIR
  maya$<$<BOOL:${OPENVDB_MAYA_SUBDIR}>:/>${MAYA_VERSION_NUMBER}
  CACHE PATH "Openvdb's maya install directory"
  )

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
  COMPILE_FLAGS "-DOPENVDB_PRIVATE"
  )

ADD_LIBRARY ( openvdb_maya SHARED
  ${OPENVDB_MAYA_MODULE_SOURCE_FILES}
  )

TARGET_LINK_LIBRARIES ( openvdb_maya
  ${OPENVDB_MAYA_DEPENDANT_LIBS}
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

IF ( OPENVDB_ENABLE_RPATH )
  # @todo There is probably a better way to do this for imported targets
  SET ( RPATHS )
  LIST ( APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${IlmBase_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
  )
  IF ( OPENVDB_BUILD_CORE )
    LIST ( APPEND ${CMAKE_INSTALL_PREFIX}/lib )
  ELSE ()
    LIST ( APPEND ${OpenVDB_LIBRARY_DIRS} )
  ENDIF ()

  LIST ( REMOVE_DUPLICATES RPATHS )

  SET_TARGET_PROPERTIES ( openvdb_maya
    PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  UNSET ( RPATHS )
ENDIF ()

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
