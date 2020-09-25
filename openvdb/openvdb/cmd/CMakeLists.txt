## Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for OpenVDB Binaries

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(OpenVDBBinaries LANGUAGES CXX)

# Monitoring <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
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

  install(TARGETS vdb_print RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
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

  install(TARGETS vdb_lod RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

##########################################################################

#### vdb_render

if(OPENVDB_BUILD_VDB_RENDER)
  find_package(IlmBase ${MINIMUM_ILMBASE_VERSION} REQUIRED COMPONENTS Half Iex IlmThread Imath)
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
    IlmBase::Imath
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

  install(TARGETS vdb_render RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

##########################################################################

#### vdb_view

if(OPENVDB_BUILD_VDB_VIEW)
  if(NOT Boost_USE_STATIC_LIBS)
    # @note  Both of these must be set for Boost 1.70 (VFX2020) to link against
    #        boost shared libraries (more specifically libraries built with -fPIC).
    #        http://boost.2283326.n4.nabble.com/CMake-config-scripts-broken-in-1-70-td4708957.html
    #        https://github.com/boostorg/boost_install/commit/160c7cb2b2c720e74463865ef0454d4c4cd9ae7c
    set(BUILD_SHARED_LIBS ON)
    set(Boost_USE_STATIC_LIBS OFF)
  endif()

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

  if(OPENVDB_ENABLE_RPATH)
    set_target_properties(vdb_view
      PROPERTIES INSTALL_RPATH "${RPATHS}"
    )
  endif()

  install(TARGETS vdb_view RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

unset(RPATHS)
