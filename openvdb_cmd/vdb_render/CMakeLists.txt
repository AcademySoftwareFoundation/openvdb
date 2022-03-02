# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for VDB Render

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)
project(VDBRender LANGUAGES CXX)

include(GNUInstallDirs)

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

set(SOURCE_FILES main.cc)
add_executable(vdb_render ${SOURCE_FILES})

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
