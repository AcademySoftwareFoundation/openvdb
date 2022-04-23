## Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Binaries

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(OpenVDBBinaries LANGUAGES CXX)

include(GNUInstallDirs)

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
  # @note  Could also use the openvdb_je target here, but we just opt to
  # handle the value of CONCURRENT_MALLOC outside of this branching for
  # both cases
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

set(OPENVDB_BINARIES_DEPENDENT_LIBS
  ${OPENVDB_LIB}
)

if(CONCURRENT_MALLOC STREQUAL "Jemalloc")
  find_package(Jemalloc REQUIRED)
  list(APPEND OPENVDB_BINARIES_DEPENDENT_LIBS Jemalloc::jemalloc)
elseif(CONCURRENT_MALLOC STREQUAL "Tbbmalloc")
  find_package(TBB ${MINIMUM_TBB_VERSION} REQUIRED COMPONENTS tbbmalloc)
  list(APPEND OPENVDB_BINARIES_DEPENDENT_LIBS TBB::tbbmalloc)
endif()

##########################################################################

##### VDB binaries

#### vdb_print

if(OPENVDB_BUILD_VDB_PRINT)
  set(VDB_PRINT_SOURCE_FILES openvdb_print.cc)
  add_executable(vdb_print ${VDB_PRINT_SOURCE_FILES})
  target_include_directories(vdb_print PRIVATE ../.)
  target_link_libraries(vdb_print ${OPENVDB_BINARIES_DEPENDENT_LIBS})

  install(TARGETS vdb_print RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

##########################################################################

#### vdb_lod

if(OPENVDB_BUILD_VDB_LOD)
  set(VDB_LOD_SOURCE_FILES  openvdb_lod.cc)
  add_executable(vdb_lod ${VDB_LOD_SOURCE_FILES})
  target_include_directories(vdb_lod PRIVATE ../.)
  target_link_libraries(vdb_lod ${OPENVDB_BINARIES_DEPENDENT_LIBS})

  install(TARGETS vdb_lod RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

##########################################################################

#### vdb_render

if(OPENVDB_BUILD_VDB_RENDER)

  if(USE_PNG)
    find_package(PNG REQUIRED)
  endif()

  if(USE_IMATH_HALF)
    find_package(Imath CONFIG)
    if (NOT TARGET Imath::Imath)
      if(USE_EXR)
        find_package(IlmBase ${MINIMUM_ILMBASE_VERSION} REQUIRED COMPONENTS Half Iex IlmThread Imath)
      else()
        find_package(IlmBase ${MINIMUM_ILMBASE_VERSION} REQUIRED COMPONENTS Half)
      endif()
    endif()
  endif()

  if(USE_EXR)
    if (NOT TARGET Imath::Imath)
      find_package(OpenEXR ${MINIMUM_OPENEXR_VERSION} REQUIRED COMPONENTS IlmImf)
    else()
      find_package(OpenEXR CONFIG REQUIRED)
    endif()
  endif()

  set(VDB_RENDER_SOURCE_FILES openvdb_render.cc)
  add_executable(vdb_render ${VDB_RENDER_SOURCE_FILES})
  target_include_directories(vdb_render PRIVATE ../.)

  if(USE_EXR)
    target_compile_definitions(vdb_render PRIVATE -DOPENVDB_USE_EXR)
  endif()
  if(USE_PNG)
    target_compile_definitions(vdb_render PRIVATE -DOPENVDB_USE_PNG)
  endif()

  # Set deps. Note that the order here is important. If we're building against
  # Houdini 17.5 we must include OpenEXR and IlmBase deps first to ensure the
  # users chosen namespaced headers are correctly prioritized. Otherwise other
  # include paths from shared installs (including houdini) may pull in the wrong
  # headers

  target_link_libraries(vdb_render
    # For Imath/OpenEXR v3.X
    $<TARGET_NAME_IF_EXISTS:Imath::Imath>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::OpenEXR>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::OpenEXRUtil>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmThread>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::Iex>
    # For IlmBase/OpenEXR v2.X
    $<TARGET_NAME_IF_EXISTS:IlmBase::Half>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmImf>
    $<TARGET_NAME_IF_EXISTS:OpenEXR::IlmImfUtil>
    $<TARGET_NAME_IF_EXISTS:IlmBase::IlmThread>
    $<TARGET_NAME_IF_EXISTS:IlmBase::Iex>
    $<TARGET_NAME_IF_EXISTS:IlmBase::Imath>
    ${OPENVDB_BINARIES_DEPENDENT_LIBS}
    $<TARGET_NAME_IF_EXISTS:PNG::PNG>
  )

  if(WIN32)
    # @note OPENVDB_OPENEXR_STATICLIB is old functionality and should be removed
    get_target_property(ILMBASE_LIB_TYPE IlmBase::Half TYPE)
    if(OPENEXR_USE_STATIC_LIBS OR (${ILMBASE_LIB_TYPE} STREQUAL STATIC_LIBRARY))
      target_compile_definitions(vdb_render PUBLIC -DOPENVDB_OPENEXR_STATICLIB)
    endif()
  endif()

  install(TARGETS vdb_render RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

##########################################################################

#### vdb_view

if(OPENVDB_BUILD_VDB_VIEW)
  find_package(OpenGL REQUIRED)

  if(WIN32)
    find_package(GLEW REQUIRED)
  endif()

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
  target_include_directories(vdb_view PRIVATE ../.)

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
endif()

unset(RPATHS)
