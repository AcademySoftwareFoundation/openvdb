# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for VDB View

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(VDBView LANGUAGES CXX)

include(GNUInstallDirs)

find_package(OpenGL REQUIRED)

if(WIN32)
  find_package(GLEW REQUIRED)
endif()

# wraps find_package(glfw3) and sets the glfw target
include(OpenVDBGLFW3Setup)

set(SOURCE_FILES
  main.cc
  Camera.cc
  ClipBox.cc
  Font.cc
  RenderModules.cc
  Viewer.cc
)

add_executable(vdb_view ${SOURCE_FILES})

target_link_libraries(vdb_view
  ${OPENVDB_BINARIES_DEPENDENT_LIBS}
  OpenGL::GL
  OpenGL::GLU
  glfw
)

if(WIN32)
  target_link_libraries(vdb_view GLEW::GLEW)
endif()

target_compile_definitions(vdb_view PRIVATE -DGL_GLEXT_PROTOTYPES=1)

install(TARGETS vdb_view RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
