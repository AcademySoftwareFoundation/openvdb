#!/usr/bin/env bash

set -ex

COMPILER="$1"; shift
RELEASE="$1"; shift
ABI="$1"; shift
BLOSC="$1"; shift
SIMD="$1"; shift
CMAKE_EXTRA="$@"

CXX_STRICT=ON
if [ $(uname -s) == "Darwin" ]; then
    echo "Disabling compiler warnings on MacOS"
    CXX_STRICT=OFF
fi

mkdir build
cd build
cmake \
    -DCMAKE_CXX_COMPILER=${COMPILER} \
    -DCMAKE_BUILD_TYPE=${RELEASE} \
    -DOPENVDB_ABI_VERSION_NUMBER=${ABI} \
    -DOPENVDB_USE_DEPRECATED_ABI=ON \
    -DUSE_BLOSC=${BLOSC} \
    -DOPENVDB_CXX_STRICT=${CXX_STRICT} \
    -DOPENVDB_BUILD_PYTHON_MODULE=ON \
    -DOPENVDB_BUILD_UNITTESTS=ON \
    -DOPENVDB_BUILD_BINARIES=ON \
    -DOPENVDB_BUILD_VDB_PRINT=ON \
    -DOPENVDB_BUILD_VDB_LOD=ON \
    -DOPENVDB_BUILD_VDB_RENDER=ON \
    -DOPENVDB_BUILD_VDB_VIEW=ON \
    -DOPENVDB_SIMD=${SIMD} \
    ${CMAKE_EXTRA} \
    ..
make -j2
