#!/usr/bin/env bash

set -ex

COMPILER="$1"; shift
RELEASE="$1"; shift
ABI="$1"; shift
BLOSC="$1"; shift
SIMD="$1"; shift
CMAKE_EXTRA="$@"

# DebugNoInfo is a custom CMAKE_BUILD_TYPE - no optimizations, no symbols, asserts enabled

mkdir -p $HOME/install
mkdir build
cd build

# print version
cmake --version

cmake \
    -DCMAKE_CXX_FLAGS_DebugNoInfo="" \
    -DCMAKE_CXX_COMPILER=${COMPILER} \
    -DCMAKE_BUILD_TYPE=${RELEASE} \
    -DOPENVDB_ABI_VERSION_NUMBER=${ABI} \
    -DOPENVDB_USE_DEPRECATED_ABI_5=ON \
    -DUSE_BLOSC=${BLOSC} \
    -DOPENVDB_BUILD_PYTHON_MODULE=ON \
    -DOPENVDB_BUILD_UNITTESTS=ON \
    -DOPENVDB_BUILD_BINARIES=ON \
    -DOPENVDB_BUILD_VDB_PRINT=ON \
    -DOPENVDB_BUILD_VDB_LOD=ON \
    -DOPENVDB_BUILD_VDB_RENDER=ON \
    -DOPENVDB_BUILD_VDB_VIEW=ON \
    -DOPENVDB_SIMD=${SIMD} \
    -DCMAKE_INSTALL_PREFIX=$HOME/install \
    ${CMAKE_EXTRA} \
    ..
make -j2
make install