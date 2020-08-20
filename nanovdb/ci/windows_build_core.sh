#!/usr/bin/env bash

set -ex

VCPKG_ROOT="$1"; shift
CMAKE_EXTRA="$@"

mkdir __build_core
cd __build_core

# See the following regarding boost installation:
#  https://github.com/actions/virtual-environments/issues/687
#  https://github.com/actions/virtual-environments/blob/master/images/win/Windows2019-Readme.md

# print version
cmake --version

cmake ..\
  -G "Visual Studio 16 2019" -A x64 \
  -DVCPKG_TARGET_TRIPLET="${VCPKG_DEFAULT_TRIPLET}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}\scripts\buildsystems\vcpkg.cmake" \
  -DNANOVDB_USE_OPENVDB=OFF \
  -DNANOVDB_USE_CUDA=ON \
  -DNANOVDB_USE_TBB=ON \
  -DNANOVDB_USE_OPENCL=ON \
  -DNANOVDB_USE_BLOSC=ON \
  -DNANOVDB_USE_ZLIB=ON \
  -DNANOVDB_USE_OPTIX=OFF \
  -DNANOVDB_BUILD_SAMPLES=ON \
  -DNANOVDB_BUILD_BENCHMARK=ON \
  -DNANOVDB_BUILD_UNITTESTS=ON \
  -DNANOVDB_BUILD_TOOLS=ON \
  -DNANOVDB_BUILD_VIEWER=ON \
  -DNANOVDB_BUILD_DOCS=OFF \
  ${CMAKE_EXTRA}

cmake --build . --parallel 4 --config Release --target install
