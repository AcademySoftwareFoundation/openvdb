## Copyright (c) DreamWorks Animation LLC
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

project(OpenVDBBinaries)
cmake_minimum_required(VERSION 3.3)
# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

###### OpenVDB Binary Component Options

option(OPENVDB_BUILD_VDB_PRINT "Build vdb_print" ON)
option(OPENVDB_BUILD_VDB_LOD "Build vdb_lod" OFF)
option(OPENVDB_BUILD_VDB_RENDER "Build vdb_render" OFF)
option(OPENVDB_BUILD_VDB_VIEW "Build vdb_view" OFF)

#########################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "----------- Configuring OpenVDBBinaries ------------")
message(STATUS "----------------------------------------------------")

##########################################################################

# Collect lib dependencies shared by all binaries

if(NOT OPENVDB_BUILD_CORE)
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

set(OPENVDB_BINARIES_DEPENDENT_LIBS
  ${OPENVDB_LIB}
)

##########################################################################

if(WIN32)
  # Because of implicit linking!
  link_directories(${Boost_LIBRARY_DIR})
  if(OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING)
    add_definitions(-DBOOST_ALL_NO_LIB)
  endif()
endif()

if(WIN32)
  add_definitions(-D_WIN32 -DNOMINMAX -DOPENVDB_DLL)
endif()

# rpath handling

set(RPATHS "")
if(OPENVDB_ENABLE_RPATH)
  # @todo There is probably a better way to do this for imported targets
  list(APPEND RPATHS
    ${Boost_LIBRARY_DIRS}
    ${IlmBase_LIBRARY_DIRS}
    ${Log4cplus_LIBRARY_DIRS}
    ${Blosc_LIBRARY_DIRS}
    ${Tbb_LIBRARY_DIRS}
  )
  if(OPENVDB_BUILD_CORE)
    list(APPEND RPATHS ${CMAKE_INSTALL_PREFIX}/lib)
  else()
    list(APPEND RPATHS ${OpenVDB_LIBRARY_DIRS})
  endif()

  list(REMOVE_DUPLICATES RPATHS)
endif()

##########################################################################

##### VDB binaries

#### vdb_print

if(OPENVDB_BUILD_VDB_PRINT)
  set(VDB_PRINT_SOURCE_FILES openvdb_print.cc)
  add_executable(vdb_print ${VDB_PRINT_SOURCE_FILES})
  target_link_libraries(vdb_print ${OPENVDB_BINARIES_DEPENDENT_LIBS})

  if(OPENVDB_ENABLE_RPATH)
    set_target_properties(vdb_print
      PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  endif()

  install(TARGETS vdb_print DESTINATION bin)
endif()

##########################################################################

#### vdb_lod

if(OPENVDB_BUILD_VDB_LOD)
  set(VDB_LOD_SOURCE_FILES  openvdb_lod.cc)
  add_executable(vdb_lod ${VDB_LOD_SOURCE_FILES})
  target_link_libraries(vdb_lod ${OPENVDB_BINARIES_DEPENDENT_LIBS})

  if(OPENVDB_ENABLE_RPATH)
    set_target_properties(vdb_lod
      PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  endif()

  install(TARGETS vdb_lod DESTINATION bin)
endif()

##########################################################################

#### vdb_render

if(OPENVDB_BUILD_VDB_RENDER)
  find_package(IlmBase ${MINIMUM_ILMBASE_VERSION} REQUIRED COMPONENTS Half Iex IlmThread)
  find_package(OpenEXR ${MINIMUM_OPENEXR_VERSION} REQUIRED COMPONENTS IlmImf)

  set(VDB_RENDER_SOURCE_FILES openvdb_render.cc)
  add_executable(vdb_render ${VDB_RENDER_SOURCE_FILES})

  # Set deps. Note that the order here is important. If we're building against
  # Houdini 17.5 we must include OpenEXR and IlmBase deps first to ensure the
  # users chosen namespaced headers are correctly prioritized. Otherwise other
  # include paths from shared installs (including houdini) may pull in the wrong
  # headers

  target_link_libraries(vdb_render
    OpenEXR::IlmImf
    IlmBase::IlmThread
    IlmBase::Iex
    ${OPENVDB_BINARIES_DEPENDENT_LIBS}
  )

  if(OPENVDB_ENABLE_RPATH)
    set(OPENVDB_RENDER_RPATHS)
    list(APPEND OPENVDB_RENDER_RPATHS
      ${OpenEXR_LIBRARY_DIRS}
      ${RPATHS}
    )
    list(REMOVE_DUPLICATES OPENVDB_RENDER_RPATHS)

    set_target_properties(vdb_render
      PROPERTIES INSTALL_RPATH "${OPENVDB_RENDER_RPATHS}"
    )
    unset(OPENVDB_RENDER_RPATHS)
  endif()

  install(TARGETS vdb_render DESTINATION bin)
endif()

##########################################################################

#### vdb_view

if(OPENVDB_BUILD_VDB_VIEW)

  # @todo improve the viewer header system

  file(GLOB VIEWER_SOURCE_FILES ../viewer/*.h)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dwa/openvdb_viewer)
  file(COPY ${VIEWER_SOURCE_FILES} DESTINATION ${CMAKE_BINARY_DIR}/dwa/openvdb_viewer)

  if(WIN32)
    message(WARNING "Currently no CMake support for building vdb_view binary on Windows.")
    # @todo
    # find_package(GLEW REQUIRED)
    # INCLUDE_DIRECTORIES ( SYSTEM ${GLEW_INCLUDE_DIR})
  else()
    find_package(Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS thread)
    find_package(OpenGL REQUIRED)

    # wraps find_package(glfw3) and sets the glfw target
    include(OpenVDBGLFW3Setup)

    set(VDB_VIEW_SOURCE_FILES
      openvdb_view.cc
      ../viewer/Camera.cc
      ../viewer/ClipBox.cc
      ../viewer/Font.cc
      ../viewer/RenderModules.cc
      ../viewer/Viewer.cc
    )

    add_executable(vdb_view ${VDB_VIEW_SOURCE_FILES})

    target_include_directories(vdb_view
      PRIVATE ${CMAKE_BINARY_DIR}/dwa
    )

    target_link_libraries(vdb_view
      ${OPENVDB_BINARIES_DEPENDENT_LIBS}
      Boost::thread
      OpenGL::GL
      OpenGL::GLU
      glfw
    )

    target_compile_definitions(vdb_view PRIVATE
      "-DOPENVDB_USE_GLFW_3" "-DGL_GLEXT_PROTOTYPES=1"
    )

    if(OPENVDB_ENABLE_RPATH)
      set_target_properties(vdb_view
        PROPERTIES INSTALL_RPATH "${RPATHS}"
      )
    endif()

    install(TARGETS vdb_view DESTINATION bin)
  endif()
endif()

unset(RPATHS)
