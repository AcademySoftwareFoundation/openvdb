#!/usr/bin/env bash

set -ex

OPENVDB_VERSION="$1"; shift
COMPILER="$1"; shift
RELEASE="$1"; shift
ABI="$1"; shift
BLOSC="$1"; shift
SIMD="$1"; shift
CMAKE_EXTRA="$@"

# DebugNoInfo is a custom CMAKE_BUILD_TYPE - no optimizations, no symbols, asserts enabled

git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb

if [ "$OPENVDB_VERSION" != "latest" ]; then
    git checkout tags/v${OPENVDB_VERSION} -b v${OPENVDB_VERSION}
fi

mkdir __build
cd __build

# print version
cmake --version

cmake \
    -DCMAKE_CXX_FLAGS_DebugNoInfo="" \
    -DCMAKE_CXX_COMPILER=${COMPILER} \
    -DCMAKE_BUILD_TYPE=${RELEASE} \
    -DOPENVDB_ABI_VERSION_NUMBER=${ABI} \
    -DOPENVDB_USE_DEPRECATED_ABI=ON \
    -DUSE_BLOSC=${BLOSC} \
    -DOPENVDB_BUILD_PYTHON_MODULE=OFF \
    -DOPENVDB_BUILD_UNITTESTS=OFF \
    -DOPENVDB_BUILD_BINARIES=OFF \
    -DOPENVDB_SIMD=${SIMD} \
    ${CMAKE_EXTRA} \
    ..
make install -j2
cd ../..
