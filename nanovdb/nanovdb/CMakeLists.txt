# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB

#]=======================================================================]

cmake_minimum_required(VERSION 3.12)
project(NanoVDB LANGUAGES C CXX)

include(GNUInstallDirs)

# option(OPENVDB_AX_SHARED "Build dynamically linked version of the ax library." ON)
# option(OPENVDB_AX_STATIC "Build statically linked version of the ax library." ON)
#
# option(OPENVDB_BUILD_AX_GRAMMAR [=[
# Rebuild the AX grammar using flex/bison. This only necessary if you are
# altering the frontend AX language, typically only a developer option. ]=] OFF)
# option(USE_NEW_PASS_MANAGER [=[
# Enable the new Optimization Pass Manager. This is a developer option.]=] OFF)
#
# mark_as_advanced(
#   USE_NEW_PASS_MANAGER
#   OPENVDB_BUILD_AX_GRAMMAR
# )

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
message(STATUS "----------------------------------------------------")
message(STATUS "----------------Configuring CXXFLAGS ---------------")
message(STATUS "----------------------------------------------------")
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
# option(NANOVDB_BUILD_UNITTESTS "Build Unit tests" OFF)
option(NANOVDB_BUILD_EXAMPLES "Build examples" ON)
# option(NANOVDB_BUILD_BENCHMARK "Build benchmark" OFF)
# option(NANOVDB_BUILD_DOCS "Build docs" OFF)
# option(NANOVDB_BUILD_TOOLS "Build command-line tools" OFF)
# option(NANOVDB_CUDA_KEEP_PTX "Keep CUDA PTX" OFF)

# option(NANOVDB_USE_INTRINSICS "Build with hardware intrinsics support" ON)
# option(NANOVDB_USE_OPENVDB "Build with OpenVDB support" ON)
# option(NANOVDB_USE_TBB "Build with TBB support" ON)
# option(NANOVDB_USE_BLOSC "Build with BLOSC support" ON)
# option(NANOVDB_USE_ZLIB "Build with ZLIB support" ON)
option(NANOVDB_USE_CUDA "Build with CUDA support" ON)
# option(NANOVDB_USE_OPENGL "Build with OpenGL support" ON)
# option(NANOVDB_USE_OPENCL "Build with OpenCL support" ON)
# option(NANOVDB_USE_OPTIX "Build with OptiX support" ON)
# option(NANOVDB_USE_MAGICAVOXEL "Build with MagicaVoxel support" ON)

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

file(MAKE_DIRECTORY NANOVDB_LOCAL_UTILDIR)
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