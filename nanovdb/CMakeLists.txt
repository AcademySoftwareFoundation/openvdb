# Copyright Contributors to the OpenVDB Project SPDX-License-Identifier: MPL-2.0

# -----------------------------------------------------------------------
# NanoVDB cmake file.

cmake_minimum_required(VERSION 3.11.4)

project(NanoVDB LANGUAGES C CXX)

# -----------------------------------------------------------------------
# Setup.

include(FetchContent)

set(CMAKE_CXX_STANDARD 11)
set(NanoVDB_TOP_LEVEL_DIR ${PROJECT_SOURCE_DIR})

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 OLD)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set the default build to Release. NB. this can't do anything for MSVC which
# defaults to Debug on startup.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
  set(USE_EMSCRIPTEN_TOOLCHAIN
      ON
      CACHE INTERNAL "using emscripten toolchain")
else()
  set(USE_EMSCRIPTEN_TOOLCHAIN
      OFF
      CACHE INTERNAL "using emscripten toolchain")
endif()

# -----------------------------------------------------------------------
option(NANOVDB_BUILD_UNITTESTS "Build Unit tests" ON)
option(NANOVDB_BUILD_EXAMPLES "Build examples" ON)
option(NANOVDB_BUILD_BENCHMARK "Build benchmark" ON)
option(NANOVDB_BUILD_DOCS "Build docs" OFF)
option(NANOVDB_BUILD_TOOLS "Build command-line tools" ON)
option(NANOVDB_CUDA_KEEP_PTX "Keep CUDA PTX" OFF)

option(NANOVDB_USE_INTRINSICS "Build with hardware intrinsics support" ON)
option(NANOVDB_USE_OPENVDB "Build with OpenVDB support" ON)
option(NANOVDB_USE_OPENGL "Build with OpenGL support" ON)
option(NANOVDB_USE_OPENCL "Build with OpenCL support" ON)
option(NANOVDB_USE_CUDA "Build with CUDA support" ON)
option(NANOVDB_USE_TBB "Build with TBB support" ON)
option(NANOVDB_USE_BLOSC "Build with BLOSC support" ON)
option(NANOVDB_USE_ZLIB "Build with ZLIB support" ON)
option(NANOVDB_USE_OPTIX "Build with OptiX support" ON)
option(NANOVDB_USE_MAGICAVOXEL "Build with MagicaVoxel support" ON)

option(NANOVDB_ALLOW_FETCHCONTENT
       "Allow FetchContent to download missing dependencies" ON)

# -----------------------------------------------------------------------
# This is required to compile OpenVDB headers on windows.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-invalid-offsetof -pthread -lpthread")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

# -----------------------------------------------------------------------
if(NANOVDB_BUILD_UNITTESTS OR NANOVDB_BUILD_BENCHMARK)
  if(NANOVDB_ALLOW_FETCHCONTENT)
    message(STATUS "Downloading GTest...")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.10.0)
    set(INSTALL_GTEST
        OFF
        CACHE BOOL "" FORCE)
    set(BUILD_GMOCK
        OFF
        CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
      FetchContent_Populate(googletest)
      add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
    endif()
    set(NANOVDB_GTEST "gtest" "gtest_main")
  else()
    find_package(GTest)
    if(GTEST_FOUND)
      set(NANOVDB_GTEST "GTest::GTest" "GTest::Main")
    endif()
  endif()
endif()

# ---------------------------------------------------------------
if(NANOVDB_USE_MAGICAVOXEL)
  if(NANOVDB_ALLOW_FETCHCONTENT)
    if(NOT ogt_POPULATED)
      message(STATUS "Downloading ogt...")

      FetchContent_Declare(
        ogt
        GIT_REPOSITORY https://github.com/jpaver/opengametools.git
        GIT_TAG master)

      FetchContent_GetProperties(ogt)
      if(NOT ogt_POPULATED)
        FetchContent_Populate(ogt)
        set(NANOVDB_OGT_INCLUDE_DIRECTORY ${ogt_SOURCE_DIR}/src)
        set(NANOVDB_USE_MAGICAVOXEL_FLAG "NANOVDB_USE_MAGICAVOXEL")
      endif()
    endif()
  endif()
endif(NANOVDB_USE_MAGICAVOXEL)

# -----------------------------------------------------------------------
if(NANOVDB_USE_CUDA AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(CUDA)
  if(CUDA_FOUND)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
    if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
      set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
    endif()

    set(CMAKE_CUDA_FLAGS
        "${NANOVDB_CUDA_EXTENDED_LAMBDA} -use_fast_math -lineinfo")

    # find NVRTC library.
    find_library(
      CUDA_nvrtc_LIBRARY nvrtc "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
      "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32")

    # workaround for win32 bug when nvcc "--keep" is used.
    if(WIN32)
      if(NANOVDB_CUDA_KEEP_PTX)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
        set(CMAKE_CUDA_FLAGS_RELEASE
            " --source-in-ptx --keep ${CMAKE_CUDA_FLAGS_RELEASE}")
      endif()
    endif()

    enable_language(CUDA)

    set(NANOVDB_USE_CUDA_FLAG "NANOVDB_USE_CUDA")
    set(CUDA_LINK_LIBRARY ${CUDA_LIBRARIES})
    set(CUDA_INCLUDE_DIRECTORY ${CUDA_INCLUDE_DIRS})
    set(NVRTC_LINK_LIBRARY ${CUDA_nvrtc_LIBRARY})

  endif()
endif()

# -----------------------------------------------------------------------
if(NANOVDB_USE_INTRINSICS)
  set(NANOVDB_USE_INTRINSICS_FLAG "NANOVDB_USE_INTRINSICS")
endif()

# -----------------------------------------------------------------------
if(NANOVDB_USE_OPENCL AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(OpenCL)
  if(OpenCL_FOUND)
    set(NANOVDB_OPENCL ${OpenCL_LIBRARIES})
    set(OPENCL_INCLUDE_DIRECTORY ${OpenCL_INCLUDE_DIRS})
    set(NANOVDB_USE_OPENCL_FLAG "NANOVDB_USE_OPENCL")
  endif()
endif()

# -----------------------------------------------------------------------
if(NANOVDB_USE_OPENGL AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(OpenGL)
  if(OpenGL_FOUND)
    set(NANOVDB_OPENGL OpenGL::GL ${CMAKE_DL_LIBS})
    set(OPENGL_INCLUDE_DIRECTORY ${OPENGL_INCLUDE_DIR})
    set(NANOVDB_USE_OPENGL_FLAG "NANOVDB_USE_OPENGL")
  endif()
endif()

# -----------------------------------------------------------------------
if(NANOVDB_USE_OPTIX AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(OptiX)
  if(OptiX_FOUND)
    set(NANOVDB_OPTIX_INCLUDE_DIRECTORY ${OptiX_INCLUDE_DIR})
    set(NANOVDB_OPTIX_LINK_LIBRARY ${OptiX_LIBRARIES})
    set(NANOVDB_USE_OPTIX_FLAG "NANOVDB_USE_OPTIX")
  endif()
endif()

# -----------------------------------------------------------------------
if(NANOVDB_USE_OPENVDB AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(OpenVDB)
  if(OpenVDB_FOUND)
    set(NANOVDB_OPENVDB "OpenVDB::openvdb")
    set(NANOVDB_USE_OPENVDB_FLAG "NANOVDB_USE_OPENVDB")
    set(NANOVDB_USE_TBB_FLAG "NANOVDB_USE_TBB")
  endif()
  # disable the shared libs that OpenVDB's cmake switches ON.
  set(BUILD_SHARED_LIBS OFF)
endif()

# -----------------------------------------------------------------------
if(NANOVDB_USE_TBB AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(TBB)
  if(Tbb_VERSION)
    set(NANOVDB_USE_TBB_FLAG "NANOVDB_USE_TBB")
    set(NANOVDB_TBB "TBB::tbb")
    if(WIN32)
      # this prevents tbb_debug.lib issue on windows
      add_definitions(-DTBB_USE_PREVIEW_BINARY)
    endif()
  endif()
endif()

# -----------------------------------------------------------------------
if(NANOVDB_USE_ZLIB AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(ZLIB REQUIRED)
  if(ZLIB_FOUND)
    set(NANOVDB_USE_ZLIB_FLAG "NANOVDB_USE_ZIP")
    set(NANOVDB_ZLIB "ZLIB::ZLIB")
  endif()
endif()

# -----------------------------------------------------------------------
if(NANOVDB_USE_BLOSC AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(Blosc REQUIRED)
  if(Blosc_FOUND)
    set(NANOVDB_USE_BLOSC_FLAG "NANOVDB_USE_BLOSC")
    set(NANOVDB_BLOSC "Blosc::blosc")
  endif()
endif()

# -----------------------------------------------------------------------
if(NOT USE_EMSCRIPTEN_TOOLCHAIN)
  # -----------------------------
  set(GLAD_SOURCE_DIR ${NanoVDB_TOP_LEVEL_DIR}/external/glad)

  list(APPEND GLAD_SOURCE_FILES ${GLAD_SOURCE_DIR}/src/glad.c
       ${GLAD_SOURCE_DIR}/include/glad/glad.h)

  add_library(glad ${GLAD_SOURCE_FILES})

  target_include_directories(glad PRIVATE ${GLAD_SOURCE_DIR}/include)

  set(NANOVDB_GLAD_LINK_LIBRARY "glad")
  set(NANOVDB_GLAD_INCLUDE_DIRECTORY ${GLAD_SOURCE_DIR}/include)
  set(NANOVDB_USE_GLAD_FLAG "NANOVDB_USE_GLAD")

endif()

# -----------------------------------------------------------------------
function(assign_source_group)
  foreach(_source IN ITEMS ${ARGN})
    if(IS_ABSOLUTE "${_source}")
      file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
    else()
      set(_source_rel "${_source}")
    endif()
    get_filename_component(_source_path "${_source_rel}" PATH)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${_source_path_msvc}" FILES "${_source}")
  endforeach()
endfunction(assign_source_group)

# -----------------------------------------------------------------------
set(NANOVDB_LOCATION ".")
set(NANOVDB_INCLUDE_DIR "${NANOVDB_LOCATION}")
set(NANOVDB_INCLUDE_DIRECTORY
    ${NANOVDB_INCLUDE_DIR}
    CACHE STRING "NANOVDB include directory")

file(GLOB_RECURSE NANOVDB_HEADERS "${NANOVDB_INCLUDE_DIR}/nanovdb/*.h")
assign_source_group(${NANOVDB_HEADERS})

# ---------------------------------------------------------------
add_library(libnanovdb INTERFACE)

target_include_directories(libnanovdb INTERFACE ${NANOVDB_INCLUDE_DIR}
                                                ${NANOVDB_INCLUDE_DIR}/nanovdb)

target_compile_definitions(libnanovdb INTERFACE "-DNOMINMAX" "-D${NANOVDB_USE_INTRINSICS_FLAG}")

add_custom_target(nanovdb SOURCES ${NANOVDB_HEADERS})

# -----------------------------------------------------------------------
# many of the sample projects depend on a data directory. This allows Debug
# launching from the cmake binary working directory.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data")

# -----------------------------------------------------------------------
if(NANOVDB_BUILD_TOOLS)
  add_subdirectory(cmd)
endif()

if(NANOVDB_BUILD_UNITTESTS)
  add_subdirectory(unittest)
endif()

if(NANOVDB_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(NANOVDB_BUILD_DOCS)
  add_subdirectory(docs)
endif()

# ---------------------------------------------------------------
# Installation
#

# ---------------------------------------------------------------
# copy header files to ${release}/nanovdb/*
install(DIRECTORY ${NANOVDB_INCLUDE_DIR}/nanovdb DESTINATION ./)

# ---------------------------------------------------------------
