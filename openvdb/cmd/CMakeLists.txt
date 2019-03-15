## Copyright (c) 2012-2019 DreamWorks Animation LLC
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

  CMake Configuration for OpenVDB Binaries

#]=======================================================================]

PROJECT ( OpenVDBBinaries )
CMAKE_MINIMUM_REQUIRED ( VERSION 3.3 )
# Monitoring <PackageName>_ROOT variables
IF ( POLICY CMP0074 )
  CMAKE_POLICY ( SET CMP0074 NEW )
ENDIF ()

###### OpenVDB Binary Component Options

OPTION ( OPENVDB_BUILD_VDB_LOD "Build vdb_lod" ON )
OPTION ( OPENVDB_BUILD_VDB_PRINT "Build vdb_print" ON )
OPTION ( OPENVDB_BUILD_VDB_RENDER "Build vdb_render" ON )
OPTION ( OPENVDB_BUILD_VDB_VIEW "Build vdb_view" ON )

#########################################################################

MESSAGE ( STATUS "----------------------------------------------------" )
MESSAGE ( STATUS "----------- Configuring OpenVDBBinaries ------------" )
MESSAGE ( STATUS "----------------------------------------------------" )

##########################################################################

# Collect and configure lib dependencies

IF ( NOT OPENVDB_BUILD_CORE )
  FIND_PACKAGE ( OpenVDB REQUIRED )
  SET ( OPENVDB_LIB OpenVDB::openvdb )
ELSE ()
  SET ( OPENVDB_LIB openvdb_shared )
  INCLUDE_DIRECTORIES ( ../../ )
ENDIF ()

FIND_PACKAGE ( ILMBase REQUIRED COMPONENTS Half Iex IlmThread )
FIND_PACKAGE ( TBB REQUIRED COMPONENTS tbb )
FIND_PACKAGE ( ZLIB REQUIRED )

IF ( USE_LOG4CPLUS )
  FIND_PACKAGE ( Log4cplus REQUIRED )
ENDIF ()

IF ( USE_BLOSC )
  FIND_PACKAGE ( Blosc ${MINIMUM_BLOSC_VERSION} REQUIRED )
ENDIF ()

FIND_PACKAGE ( Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS iostreams system )

IF ( UNIX )
  FIND_PACKAGE ( Threads REQUIRED )
ENDIF ()

# Set core deps. Note that the order here is important. If we're building against
# Houdini 17.5 we must include OpenEXR and ILMBase deps first to ensure the normal
# namespaced headers are used over the Houdini 17.5 ones (which will be imported
# through targets like tbb which are still deployed with Houdini)

SET ( OPENVDB_BINARIES_DEPENDANT_LIBS
  ${OPENVDB_LIB}
  Boost::iostreams
  Boost::system
  ILMBase::Half
  )

IF ( USE_LOG4CPLUS )
  LIST ( APPEND OPENVDB_BINARIES_DEPENDANT_LIBS Log4cplus::log4cplus )
ENDIF ()

IF ( USE_BLOSC )
  LIST ( APPEND OPENVDB_BINARIES_DEPENDANT_LIBS Blosc::blosc )
ENDIF ()

LIST ( APPEND OPENVDB_BINARIES_DEPENDANT_LIBS
  TBB::tbb
  ZLIB::ZLIB
  Threads::Threads
  m
  stdc++
)

##########################################################################

IF ( WIN32 )
  # Because of implicit linking!
  LINK_DIRECTORIES ( ${Boost_LIBRARY_DIR} )
  IF ( OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING )
    ADD_DEFINITIONS ( -DBOOST_ALL_NO_LIB )
  ENDIF ()
ENDIF ()

IF ( WIN32 )
  ADD_DEFINITIONS ( -D_WIN32 -DNOMINMAX -DHALF_EXPORTS -DOPENVDB_DLL )
ELSE ()
  ADD_DEFINITIONS ( -fPIC )
ENDIF ()

# rpath handling

SET ( RPATHS )
IF ( OPENVDB_ENABLE_RPATH )
  # @todo There is probably a better way to do this for imported targets
  LIST ( APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${ILMBase_LIBRARY_DIRS}
    ${Log4cplus_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
    ${Tbb_LIBRARY_DIRS}
  )
  IF ( OPENVDB_BUILD_CORE )
    LIST ( APPEND RPATHS ${CMAKE_INSTALL_PREFIX}/lib )
  ELSE ()
    LIST ( APPEND RPATHS ${OpenVDB_LIBRARY_DIRS} )
  ENDIF ()

  LIST ( REMOVE_DUPLICATES RPATHS )
ENDIF ()

# @todo improve the viewer header system

FILE ( GLOB VIEWER_SOURCE_FILES ../viewer/*.h )
FILE ( MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dwa/openvdb_viewer )
FILE ( COPY ${VIEWER_SOURCE_FILES} DESTINATION ${CMAKE_BINARY_DIR}/dwa/openvdb_viewer )
INCLUDE_DIRECTORIES ( ${CMAKE_BINARY_DIR}/dwa )

##########################################################################

##### VDB binaries

#### vdb_lod

IF ( OPENVDB_BUILD_VDB_LOD )
  SET ( VDB_LOD_SOURCE_FILES  openvdb_lod.cc )
  ADD_EXECUTABLE ( vdb_lod ${VDB_LOD_SOURCE_FILES} )
  TARGET_LINK_LIBRARIES ( vdb_lod ${OPENVDB_BINARIES_DEPENDANT_LIBS} )

  IF ( USE_LOG4CPLUS )
    TARGET_COMPILE_DEFINITIONS ( vdb_lod PRIVATE  "-DOPENVDB_USE_LOG4CPLUS" )
  ENDIF ()

  IF ( OPENVDB_ENABLE_RPATH )
    SET_TARGET_PROPERTIES ( vdb_lod
      PROPERTIES INSTALL_RPATH "${RPATHS}"
      )
  ENDIF ()

  INSTALL ( TARGETS vdb_lod DESTINATION bin )
ENDIF ()

##########################################################################

#### vdb_print

IF ( OPENVDB_BUILD_VDB_PRINT )
  SET ( VDB_PRINT_SOURCE_FILES openvdb_print.cc )
  ADD_EXECUTABLE ( vdb_print ${VDB_PRINT_SOURCE_FILES} )
  TARGET_LINK_LIBRARIES ( vdb_print ${OPENVDB_BINARIES_DEPENDANT_LIBS} )
  IF ( USE_LOG4CPLUS )
    TARGET_COMPILE_DEFINITIONS ( vdb_print PRIVATE "-DOPENVDB_USE_LOG4CPLUS" )
  ENDIF ()

  IF ( OPENVDB_ENABLE_RPATH )
    SET_TARGET_PROPERTIES ( vdb_print
      PROPERTIES INSTALL_RPATH "${RPATHS}"
      )
  ENDIF ()

  INSTALL ( TARGETS vdb_print DESTINATION bin )
ENDIF ()

##########################################################################

#### vdb_render

IF ( OPENVDB_BUILD_VDB_RENDER )
  FIND_PACKAGE ( OpenEXR REQUIRED COMPONENTS IlmImf )

  SET ( VDB_RENDER_SOURCE_FILES openvdb_render.cc )
  ADD_EXECUTABLE ( vdb_render ${VDB_RENDER_SOURCE_FILES} )
  TARGET_LINK_LIBRARIES ( vdb_render
    OpenEXR::IlmImf
    ILMBase::IlmThread
    ILMBase::Iex
    ${OPENVDB_BINARIES_DEPENDANT_LIBS} # include last - see comment above
    )
  IF ( USE_LOG4CPLUS )
    TARGET_COMPILE_DEFINITIONS ( vdb_render PRIVATE
      "-DOPENVDB_USE_LOG4CPLUS"
      )
  ENDIF ()

  IF ( OPENVDB_ENABLE_RPATH )
    SET ( OPENVDB_RENDER_RPATHS )
    LIST ( APPEND OPENVDB_RENDER_RPATHS
      ${OpenEXR_LIBRARY_DIRS}
      ${RPATHS}
    )
    LIST ( REMOVE_DUPLICATES OPENVDB_RENDER_RPATHS )

    SET_TARGET_PROPERTIES ( vdb_render
      PROPERTIES INSTALL_RPATH "${OPENVDB_RENDER_RPATHS}"
      )
    UNSET ( OPENVDB_RENDER_RPATHS )
  ENDIF ()

  INSTALL ( TARGETS vdb_render DESTINATION bin )
ENDIF ()

##########################################################################

#### vdb_view

IF ( OPENVDB_BUILD_VDB_VIEW )
  IF ( WIN32 )
    MESSAGE ( WARNING "Currently no CMake support for building vdb_view binary on Windows." )
    # @todo
    # FIND_PACKAGE ( GLEW REQUIRED )
    # INCLUDE_DIRECTORIES ( SYSTEM ${GLEW_INCLUDE_DIR} )
  ELSE ()
    FIND_PACKAGE ( Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS thread )
    FIND_PACKAGE ( OpenGL REQUIRED )

    # wraps FIND_PACKAGE ( glfw3 ) and sets the glfw target
    INCLUDE ( OpenVDBGLFW3Setup )

    SET ( VDB_VIEW_SOURCE_FILES
      openvdb_view.cc
      ../viewer/Camera.cc
      ../viewer/ClipBox.cc
      ../viewer/Font.cc
      ../viewer/RenderModules.cc
      ../viewer/Viewer.cc
      )

    ADD_EXECUTABLE ( vdb_view ${VDB_VIEW_SOURCE_FILES} )
    TARGET_LINK_LIBRARIES ( vdb_view
      Boost::thread
      OpenGL::GL
      OpenGL::GLU
      glfw
      dl
      ${OPENVDB_BINARIES_DEPENDANT_LIBS} # include last - see comment above
      )

    TARGET_COMPILE_DEFINITIONS ( vdb_view PRIVATE
      "-DOPENVDB_USE_GLFW_3" "-DGL_GLEXT_PROTOTYPES=1"
      )
    IF ( USE_LOG4CPLUS )
      TARGET_COMPILE_DEFINITIONS ( vdb_view PRIVATE
        "-DOPENVDB_USE_LOG4CPLUS"
        )
    ENDIF ()

    IF ( OPENVDB_ENABLE_RPATH )
      SET_TARGET_PROPERTIES ( vdb_view
        PROPERTIES INSTALL_RPATH "${RPATHS}"
        )
    ENDIF ()

    INSTALL ( TARGETS vdb_view DESTINATION bin )
  ENDIF ()
ENDIF ()

UNSET ( RPATHS )
