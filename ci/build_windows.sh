#!/usr/bin/env bash

set -ex

BUILD_TYPE="$1"; shift
CMAKE_EXTRA="$@"

mkdir build
cd build

# print version
cmake --version

cmake \
  -G "Visual Studio 16 2019" -A x64 \
  -DVCPKG_TARGET_TRIPLET="${VCPKG_DEFAULT_TRIPLET}" \
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_INSTALLATION_ROOT}\scripts\buildsystems\vcpkg.cmake" \
  -DOPENVDB_BUILD_BINARIES=ON \
  -DOPENVDB_BUILD_VDB_PRINT=ON \
  -DOPENVDB_BUILD_VDB_LOD=ON \
  -DOPENVDB_BUILD_VDB_RENDER=ON \
  -DOPENVDB_BUILD_VDB_VIEW=ON \
  -DOPENVDB_BUILD_UNITTESTS=ON \
  -DMSVC_MP_THREAD_COUNT=4 \
  ${CMAKE_EXTRA} \
  ..

# NOTE: --parallel only effects the number of projects build, not t-units.
# We support this with out own MSVC_MP_THREAD_COUNT option.
# Alternatively it is mentioned that the following should work:
#   cmake --build . --  /p:CL_MPcount=8
# However it does not seem to for our project.
# https://gitlab.kitware.com/cmake/cmake/-/issues/20564

cmake --build . --parallel 4 --config $BUILD_TYPE --target install
