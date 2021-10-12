# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(NanoVDB LANGUAGES C CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "--------------- Configuring NanoVDB ----------------")
message(STATUS "----------------------------------------------------")

###############################################################################
# TODO: set this option if you want to build something in nano that depends
# on openvdb
if(NOT OPENVDB_BUILD_CORE)
  set(OPENVDB_LIB OpenVDB::openvdb)
else()
  set(OPENVDB_LIB openvdb)
endif()

###############################################################################
# Configuring CXXFLAGS
###############################################################################
# This is required to compile OpenVDB headers on windows.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-invalid-offsetof -pthread -lpthread")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

###############################################################################
# add options
###############################################################################
option(NANOVDB_BUILD_UNITTESTS "Build Unit tests" ON)
option(NANOVDB_BUILD_EXAMPLES "Build examples" ON)
option(NANOVDB_BUILD_BENCHMARK "Build benchmark" ON)
option(NANOVDB_BUILD_DOCS "Build docs" OFF)
option(NANOVDB_BUILD_TOOLS "Build command-line tools" ON)
option(NANOVDB_CUDA_KEEP_PTX "Keep CUDA PTX" ON)

option(NANOVDB_USE_INTRINSICS "Build with hardware intrinsics support" ON)
option(NANOVDB_USE_OPENVDB "Build with OpenVDB support" ON)
option(NANOVDB_USE_TBB "Build with TBB support" ON)
option(NANOVDB_USE_BLOSC "Build with BLOSC support" ON)
option(NANOVDB_USE_ZLIB "Build with ZLIB support" ON)
option(NANOVDB_USE_CUDA "Build with CUDA support" ON)
option(NANOVDB_USE_OPENGL "Build with OpenGL support" ON)
option(NANOVDB_USE_OPENCL "Build with OpenCL support" ON)
option(NANOVDB_USE_OPTIX "Build with OptiX support" ON)
option(NANOVDB_USE_MAGICAVOXEL "Build with MagicaVoxel support" OFF)

if(NANOVDB_USE_CUDA AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(CUDA REQUIRED)
  if(CUDA_FOUND)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
    if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
      set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
    endif()

    set(CMAKE_CUDA_FLAGS
        "${NANOVDB_CUDA_EXTENDED_LAMBDA} -use_fast_math -lineinfo")

    # TODO: double check this with Ken
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>")

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

if(NANOVDB_USE_INTRINSICS)
  set(NANOVDB_USE_INTRINSICS_FLAG "NANOVDB_USE_INTRINSICS")
endif()

#-------------------------------------------------------------------------------
if(NANOVDB_USE_OPENCL AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(OpenCL)
  if(OpenCL_FOUND)
    set(NANOVDB_OPENCL ${OpenCL_LIBRARIES})
    set(OPENCL_INCLUDE_DIRECTORY ${OpenCL_INCLUDE_DIRS})
    set(NANOVDB_USE_OPENCL_FLAG "NANOVDB_USE_OPENCL")
  endif()
endif()

#-------------------------------------------------------------------------------
if(NANOVDB_USE_OPENGL AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(OpenGL)
  if(OpenGL_FOUND)
    set(NANOVDB_OPENGL OpenGL::GL ${CMAKE_DL_LIBS})
    set(OPENGL_INCLUDE_DIRECTORY ${OPENGL_INCLUDE_DIR})
    set(NANOVDB_USE_OPENGL_FLAG "NANOVDB_USE_OPENGL")
  endif()
endif()

#-------------------------------------------------------------------------------
if(NANOVDB_USE_OPTIX AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(OptiX)
  if(OptiX_FOUND)
    set(NANOVDB_OPTIX_INCLUDE_DIRECTORY ${OptiX_INCLUDE_DIR})
    set(NANOVDB_OPTIX_LINK_LIBRARY ${OptiX_LIBRARIES})
    set(NANOVDB_USE_OPTIX_FLAG "NANOVDB_USE_OPTIX")
  endif()
endif()

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
if(NANOVDB_USE_ZLIB AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(ZLIB REQUIRED)
  if(ZLIB_FOUND)
    set(NANOVDB_USE_ZLIB_FLAG "NANOVDB_USE_ZIP")
    set(NANOVDB_ZLIB "ZLIB::ZLIB")
  endif()
endif()

#-------------------------------------------------------------------------------
if(NANOVDB_USE_BLOSC AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
  find_package(Blosc REQUIRED)
  if(Blosc_FOUND)
    set(NANOVDB_USE_BLOSC_FLAG "NANOVDB_USE_BLOSC")
    set(NANOVDB_BLOSC "Blosc::blosc")
  endif()
endif()

###############################################################################
# Installation
###############################################################################
set(NANOVDB_LOCAL_INCLUDEDIR ${PROJECT_BINARY_DIR}/nanovdb)
set(NANOVDB_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/nanovdb)
set(NANOVDB_LOCAL_UTILDIR ${NANOVDB_LOCAL_INCLUDEDIR}/util)
set(NANOVDB_INSTALL_UTILDIR ${NANOVDB_INSTALL_INCLUDEDIR}/util)

# NanoVDB header files
set(NANOVDB_INCLUDE_FILES
  CNanoVDB.h
  NanoVDB.h
  PNanoVDB.h
)

# NanoVDB util header files
set(NANOVDB_INCLUDE_UTILFILES
  util/CSampleFromVoxels.h
  util/CudaDeviceBuffer.h
  util/DitherLUT.h
  util/ForEach.h
  util/GridBuilder.h
  util/GridChecksum.h
  util/GridHandle.h
  util/GridStats.h
  util/GridValidator.h
  util/HDDA.h
  util/HostBuffer.h
  util/Invoke.h
  util/IO.h
  util/NanoToOpenVDB.h
  util/NodeManager.h
  util/OpenToNanoVDB.h
  util/Primitives.h
  util/Range.h
  util/Ray.h
  util/Reduce.h
  util/SampleFromVoxels.h
  util/Stencils.h
)

add_library(nanovdb INTERFACE)
target_include_directories(nanovdb INTERFACE
  ${NANOVDB_LOCAL_INCLUDEDIR}
  ${NANOVDB_LOCAL_UTILDIR}
  ${PROJECT_BINARY_DIR}
)
target_compile_definitions(nanovdb INTERFACE "-DNOMINMAX" "-D${NANOVDB_USE_INTRINSICS_FLAG}")

file(MAKE_DIRECTORY ${NANOVDB_LOCAL_INCLUDEDIR})
file(COPY ${NANOVDB_INCLUDE_FILES} DESTINATION ${NANOVDB_LOCAL_INCLUDEDIR})
install(FILES ${NANOVDB_INCLUDE_FILES} DESTINATION ${NANOVDB_INSTALL_INCLUDEDIR})

file(MAKE_DIRECTORY ${NANOVDB_LOCAL_UTILDIR})
file(COPY ${NANOVDB_INCLUDE_UTILFILES} DESTINATION ${NANOVDB_LOCAL_UTILDIR})
install(FILES ${NANOVDB_INCLUDE_UTILFILES} DESTINATION ${NANOVDB_INSTALL_UTILDIR})

###############################################################################
# Options
###############################################################################
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