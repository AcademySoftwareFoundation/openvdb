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
#[=======================================================================[

  CMake Configuration for OpenVDB Maya

#]=======================================================================]

project(OpenVDBMaya)
cmake_minimum_required(VERSION 3.3)
# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

###### OpenVDB Maya Options

option(OPENVDB_MAYA_INSTALL_MOD "Install module file" ON)

if(OPENVDB_MAYA_SUBDIR)
  message(DEPRECATION "OPENVDB_MAYA_SUBDIR is deprecated and will be removed. "
    "Set OPENVDB_MAYA_INSTALL_PREFIX to your install location."
  )
endif()

##########################################################################

message(STATUS "----------------------------------------------------" )
message(STATUS "------------- Configuring OpenVDBMaya --------------" )
message(STATUS "----------------------------------------------------" )

#########################################################################

# Configure prefix

if(NOT OPENVDB_MAYA_INSTALL_PREFIX)
  if(OPENVDB_MAYA_SUBDIR)
    set(MAYA_INSTALL_LOCATION
      ${CMAKE_INSTALL_PREFIX}/maya/${Maya_VERSION}
    )
  else()
    set(MAYA_INSTALL_LOCATION
      ${CMAKE_INSTALL_PREFIX}/maya${Maya_VERSION}
    )
  endif()
endif()

set(OPENVDB_MAYA_INSTALL_PREFIX ${MAYA_INSTALL_LOCATION}
  CACHE PATH "Base install path for OpenVDB Maya libraries and headers."
)

message(STATUS "Maya base install path: ${OPENVDB_MAYA_INSTALL_PREFIX}")

#########################################################################

# Collect and configure lib dependencies

if(NOT OPENVDB_BUILD_CORE)
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

find_package(OpenGL REQUIRED)

set(OPENVDB_MAYA_DEPENDENT_LIBS
  ${OPENVDB_LIB}
  Maya
  OpenGL::GL
)

#########################################################################

# WARNING:  Compilation order of these files is extremely important. On
# MacOS/Clang (and potentially other platform/compiler combos), TBB fails
# to mutex lock the node registry in OpenVDBPlugin.cc if OpenVDBPlugin.cc
# is linked in after any of the nodes which create NodeRegistry objects.
# This is most likely due to the construction order of file static data.
# @todo  Restructure the node registry implementation

set(OPENVDB_MAYA_MODULE_SOURCE_FILES
  OpenVDBData.cc
  OpenVDBPlugin.cc
  OpenVDBCopyNode.cc
  OpenVDBFilterNode.cc
  OpenVDBFromMayaFluidNode.cc
  OpenVDBFromPolygonsNode.cc
  OpenVDBReadNode.cc
  OpenVDBToPolygonsNode.cc
  OpenVDBTransformNode.cc
  OpenVDBUtil.cc
  OpenVDBVisualizeNode.cc
  OpenVDBWriteNode.cc
)

set(OPENVDB_MAYA_MODULE_MEL_FILES
  AEOpenVDBCopyTemplate.mel
  AEOpenVDBFilterTemplate.mel
  AEOpenVDBFromMayaFluidTemplate.mel
  AEOpenVDBFromPolygonsTemplate.mel
  AEOpenVDBReadTemplate.mel
  AEOpenVDBToPolygonsTemplate.mel
  AEOpenVDBTransformTemplate.mel
  AEOpenVDBVisualizeTemplate.mel
  AEOpenVDBWriteTemplate.mel
)

#########################################################################

# This is to work around DWA way of doing stuff
set(OPENVDB_MAYA_HEADERS
  "OpenVDBData.h"
  "OpenVDBUtil.h"
)

set(OPENVDB_MAYA_DIR ${PROJECT_BINARY_DIR}/openvdb_maya)
file(MAKE_DIRECTORY ${OPENVDB_MAYA_DIR})
file(COPY ${OPENVDB_MAYA_HEADERS} DESTINATION ${OPENVDB_MAYA_DIR})

#########################################################################

add_library(openvdb_maya SHARED
  ${OPENVDB_MAYA_MODULE_SOURCE_FILES}
)

target_include_directories(openvdb_maya PUBLIC ${PROJECT_BINARY_DIR})
target_link_libraries(openvdb_maya ${OPENVDB_MAYA_DEPENDENT_LIBS})

maya_set_library_properties(openvdb_maya)

set_target_properties(openvdb_maya
  PROPERTIES
  OUTPUT_NAME OpenVDB
)

if(OPENVDB_ENABLE_RPATH)
  set(RPATHS "")
  if(OPENVDB_BUILD_CORE)
    list(APPEND RPATHS ${CMAKE_INSTALL_PREFIX}/lib)
  else()
    list(APPEND RPATHS ${OpenVDB_LIBRARY_DIRS})
  endif()

  list(REMOVE_DUPLICATES RPATHS)
  set_target_properties(openvdb_maya
    PROPERTIES INSTALL_RPATH "${RPATHS}"
  )
  unset(RPATHS)
endif()

#########################################################################

install(TARGETS
  openvdb_maya
  DESTINATION
  ${OPENVDB_MAYA_INSTALL_PREFIX}/plug-ins
)

install(FILES
  ${OPENVDB_MAYA_MODULE_MEL_FILES}
  DESTINATION
  ${OPENVDB_MAYA_INSTALL_PREFIX}/scripts
)

if(OPENVDB_MAYA_INSTALL_MOD)
  install(CODE
    "FILE (WRITE ${OPENVDB_MAYA_INSTALL_PREFIX}/openvdb.mod \"+ openvdb any ${OPENVDB_MAYA_INSTALL_PREFIX}\n\")"
    COMPONENT Runtime
  )
endif()
