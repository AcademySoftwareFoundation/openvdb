# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
project(NanoVDB LANGUAGES C CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "--------------- Configuring NanoVDB ----------------")
message(STATUS "----------------------------------------------------")

###############################################################################
# add options
###############################################################################

option(NANOVDB_BUILD_TOOLS "Build command-line tools" ON)
option(NANOVDB_BUILD_UNITTESTS "Build Unit tests" OFF)
option(NANOVDB_BUILD_EXAMPLES "Build examples" OFF)
#option(NANOVDB_BUILD_BENCHMARK "Build benchmark in examples" OFF)

option(NANOVDB_USE_INTRINSICS "Build with hardware intrinsics support" OFF)
option(NANOVDB_USE_CUDA "Build with CUDA support" OFF)
option(NANOVDB_CUDA_KEEP_PTX "Keep CUDA PTX" OFF)

option(NANOVDB_USE_OPENVDB "Build with OpenVDB support" OFF)
option(NANOVDB_USE_BLOSC "Build with BLOSC support" ${USE_BLOSC})
option(NANOVDB_USE_ZLIB "Build with ZLIB support" ${USE_ZLIB})
option(NANOVDB_USE_TBB "Build with TBB support" ${USE_TBB})
option(NANOVDB_USE_MAGICAVOXEL "Build with MagicaVoxel support" OFF)

option(NANOVDB_ALLOW_FETCHCONTENT
  "Allow FetchContent to download missing dependencies" OFF)


###############################################################################

# sanitize user input

if(NANOVDB_USE_OPENVDB)
  if(NOT NANOVDB_USE_TBB)
    message(FATAL_ERROR "Invalid CMake build configuration:
      NANOVDB_USE_OPENVDB : [ON]
      NANOVDB_USE_TBB     : [OFF]
    If you are linking NanoVDB against OpenVDB, then you
    need to use TBB, (i.e. -DNANOVDB_USE_TBB=ON).")
  endif()
endif()


###############################################################################

enable_testing()

# Add our cmake modules

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")

if(UNIX)
  # For CMake's find Threads module which brings in pthread - This flag
  # forces the compiler -pthread flag vs -lpthread
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
endif()

#if(NANOVDB_BUILD_UNITTESTS OR NANOVDB_BUILD_BENCHMARK)
if(NANOVDB_BUILD_UNITTESTS)
  find_package(GTest REQUIRED)
endif()

if(NANOVDB_USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    # Allow the user to provide CMAKE_CUDA_ARCHITECTURES
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
  endif()

  enable_language(CUDA)

  set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
  if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
    set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
  endif()

  set(CMAKE_CUDA_FLAGS "${NANOVDB_CUDA_EXTENDED_LAMBDA} -use_fast_math -lineinfo ${CMAKE_CUDA_FLAGS}")

  # workaround for win32 bug when nvcc "--keep" is used.
  if(WIN32)
    if(NANOVDB_CUDA_KEEP_PTX)
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
      set(CMAKE_CUDA_FLAGS_RELEASE
          " --source-in-ptx --keep ${CMAKE_CUDA_FLAGS_RELEASE}")
    endif()
  endif()

  if(MSVC)
    # If a CMAKE_MSVC_RUNTIME_LIBRARY has not been provided and we're building
    # against openvdb, match the CRT that openvdb has been built against. This
    # setting is automatically propagated to CXX files but not CUDA sources, so
    # the VDB_MSVC_RUNTIME_SELECTION variables is used on targets with .cu files
    if(NOT CMAKE_MSVC_RUNTIME_LIBRARY AND NANOVDB_USE_OPENVDB AND OPENVDB_BUILD_CORE)
      get_target_property(VDB_MSVC_RUNTIME_SELECTION openvdb MSVC_RUNTIME_LIBRARY)
    endif()
  endif()
endif()

if(NANOVDB_USE_OPENVDB)
  if(NOT OPENVDB_BUILD_CORE AND NOT TARGET OpenVDB::openvdb)
    find_package(OpenVDB REQUIRED COMPONENTS openvdb)
  endif()
endif()

if(NANOVDB_USE_TBB AND NOT TARGET TBB::tbb)
  find_package(TBB REQUIRED)
endif()

if(NANOVDB_USE_BLOSC AND NOT TARGET Blosc::blosc)
  find_package(Blosc REQUIRED)
endif()

if(NANOVDB_USE_ZLIB AND NOT TARGET ZLIB::ZLIB)
  find_package(ZLIB REQUIRED)
endif()

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
      endif()
    endif()
  endif()
endif()

###############################################################################
# Installation
###############################################################################

# NanoVDB header files
set(NANOVDB_INCLUDE_FILES
  CNanoVDB.h
  GridHandle.h
  HostBuffer.h
  NanoVDB.h
  NodeManager.h
  PNanoVDB.h
)

# NanoVDB cuda header files
set(NANOVDB_INCLUDE_CUDA_FILES
  cuda/DeviceBuffer.h
  cuda/GridHandle.cuh
  cuda/NodeManager.cuh
)

# NanoVDB io header files
set(NANOVDB_INCLUDE_IO_FILES
  io/IO.h
)

# NanoVDB math header files
set(NANOVDB_INCLUDE_MATH_FILES
  math/CSampleFromVoxels.h
  math/DitherLUT.h
  math/HDDA.h
  math/Math.h
  math/Ray.h
  math/SampleFromVoxels.h
  math/Stencils.h
)

# NanoVDB tools header files
set(NANOVDB_INCLUDE_TOOLS_FILES
  tools/CreateNanoGrid.h
  tools/CreatePrimitives.h
  tools/GridBuilder.h
  tools/GridChecksum.h
  tools/GridStats.h
  tools/GridValidator.h
  tools/NanoToOpenVDB.h
)

# NanoVDB tools/cuda header files
set(NANOVDB_INCLUDE_TOOLS_CUDA_FILES
  tools/cuda/AddBlindData.cuh
  tools/cuda/GridChecksum.cuh
  tools/cuda/GridStats.cuh
  tools/cuda/GridValidator.cuh
  tools/cuda/IndexToGrid.cuh
  tools/cuda/PointsToGrid.cuh
  tools/cuda/SignedFloodFill.cuh
)

# NanoVDB util header files
set(NANOVDB_INCLUDE_UTIL_FILES
  util/CpuTimer.h
  util/CreateNanoGrid.h
  util/DitherLUT.h
  util/ForEach.h
  util/GridBuilder.h
  util/GridChecksum.h
  util/GridStats.h
  util/GridValidator.h
  util/HDDA.h
  util/HostBuffer.h
  util/Invoke.h
  util/IO.h
  util/NanoToOpenVDB.h
  util/NodeManager.h
  util/OpenToNanoVDB.h
  util/PrefixSum.h
  util/Primitives.h
  util/Range.h
  util/Ray.h
  util/Reduce.h
  util/SampleFromVoxels.h
  util/Stencils.h
  util/Timer.h
  util/Util.h
)

# NanoVDB util/cuda header files
set(NANOVDB_INCLUDE_UTIL_CUDA_FILES
  util/cuda/CudaAddBlindData.cuh
  util/cuda/CudaGridHandle.cuh
  util/cuda/CudaIndexToGrid.cuh
  util/cuda/CudaSignedFloodFill.cuh
  util/cuda/Timer.h
  util/cuda/CudaDeviceBuffer.h
  util/cuda/CudaGridStats.cuh
  util/cuda/CudaNodeManager.cuh
  util/cuda/CudaUtils.h
  util/cuda/Util.h
  util/cuda/CudaGridChecksum.cuh
  util/cuda/CudaGridValidator.cuh
  util/cuda/CudaPointsToGrid.cuh
  util/cuda/GpuTimer.h
)

add_library(nanovdb INTERFACE)
target_include_directories(nanovdb INTERFACE ../)
target_compile_options(nanovdb INTERFACE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-invalid-offsetof>"
  "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>")

if(WIN32)
  target_compile_definitions(nanovdb INTERFACE -DNOMINMAX -D_USE_MATH_DEFINES)
endif()

if(NANOVDB_USE_INTRINSICS)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_INTRINSICS)
endif()

if(NANOVDB_USE_OPENVDB)
  if(NOT OPENVDB_BUILD_CORE)
    target_link_libraries(nanovdb INTERFACE OpenVDB::openvdb)
  else()
    target_link_libraries(nanovdb INTERFACE openvdb)
  endif()
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_OPENVDB)
endif()

if(NANOVDB_USE_TBB)
  target_link_libraries(nanovdb INTERFACE TBB::tbb)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_TBB)
  if(WIN32)
    # this prevents tbb_debug.lib issue on windows
    target_compile_definitions(nanovdb INTERFACE -DTBB_USE_PREVIEW_BINARY)
  endif()
endif()

if(NANOVDB_USE_BLOSC)
  target_link_libraries(nanovdb INTERFACE Blosc::blosc)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_BLOSC)
endif()

if(NANOVDB_USE_ZLIB)
  target_link_libraries(nanovdb INTERFACE ZLIB::ZLIB)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_ZIP)
endif()

if(NANOVDB_USE_CUDA)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    target_compile_options(nanovdb INTERFACE
      "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-arch=sm_75>")
  endif()

  # Add CUDA includes to any C++ units which use NanoVDB
  target_include_directories(nanovdb INTERFACE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(NANOVDB_USE_MAGICAVOXEL)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_MAGICAVOXEL)
  target_include_directories(nanovdb INTERFACE ${NANOVDB_OGT_INCLUDE_DIRECTORY})
endif()

if(TARGET Threads::Threads)
  target_link_libraries(nanovdb INTERFACE Threads::Threads)
endif()

set(NANOVDB_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/nanovdb)
set(NANOVDB_INSTALL_CUDA_DIR ${NANOVDB_INSTALL_INCLUDE_DIR}/cuda)
set(NANOVDB_INSTALL_IO_DIR ${NANOVDB_INSTALL_INCLUDE_DIR}/io)
set(NANOVDB_INSTALL_MATH_DIR ${NANOVDB_INSTALL_INCLUDE_DIR}/math)
set(NANOVDB_INSTALL_TOOLS_DIR ${NANOVDB_INSTALL_INCLUDE_DIR}/tools)
set(NANOVDB_INSTALL_TOOLS_CUDA_DIR ${NANOVDB_INSTALL_TOOLS_DIR}/cuda)
set(NANOVDB_INSTALL_UTIL_DIR ${NANOVDB_INSTALL_INCLUDE_DIR}/util)
set(NANOVDB_INSTALL_UTIL_CUDA_DIR ${NANOVDB_INSTALL_UTIL_DIR}/cuda)

install(FILES ${NANOVDB_INCLUDE_FILES} DESTINATION ${NANOVDB_INSTALL_INCLUDE_DIR})
install(FILES ${NANOVDB_INCLUDE_CUDA_FILES} DESTINATION ${NANOVDB_INSTALL_CUDA_DIR})
install(FILES ${NANOVDB_INCLUDE_IO_FILES} DESTINATION ${NANOVDB_INSTALL_IO_DIR})
install(FILES ${NANOVDB_INCLUDE_MATH_FILES} DESTINATION ${NANOVDB_INSTALL_MATH_DIR})
install(FILES ${NANOVDB_INCLUDE_TOOLS_FILES} DESTINATION ${NANOVDB_INSTALL_TOOLS_DIR})
install(FILES ${NANOVDB_INCLUDE_TOOLS_CUDA_FILES} DESTINATION ${NANOVDB_INSTALL_TOOLS_CUDA_DIR})
install(FILES ${NANOVDB_INCLUDE_UTIL_FILES} DESTINATION ${NANOVDB_INSTALL_UTIL_DIR})
install(FILES ${NANOVDB_INCLUDE_UTIL_CUDA_FILES} DESTINATION ${NANOVDB_INSTALL_UTIL_CUDA_DIR})

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
