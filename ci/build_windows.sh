#!/usr/bin/env bash

set -ex

VCPKG_ROOT="$1"

mkdir build
cd build
cmake \
  -G "Visual Studio 16 2019" -A x64 \
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}\scripts\buildsystems\vcpkg.cmake" \
  -DOPENVDB_BUILD_BINARIES=ON \
  -DOPENVDB_BUILD_VDB_PRINT=ON \
  -DOPENVDB_BUILD_VDB_LOD=ON \
  -DOPENVDB_BUILD_VDB_RENDER=ON \
  -DOPENVDB_BUILD_VDB_VIEW=ON \
  -DOPENVDB_BUILD_UNITTESTS=ON \
  ..
cmake --build . --parallel 4 --config Release --target install
