#!/usr/bin/env bash

set -ex

COMPILER="$1"; shift
NAME="$1"; shift
RELEASE="$1"; shift
CMAKE_EXTRA="$@"

mkdir -p __build_core/${NAME}
cd __build_core/${NAME}

# print version
cmake --version

cmake \
	-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_INSTALL_PREFIX="../../__release_core/${NAME}" \
    -DCMAKE_CXX_COMPILER=${COMPILER} \
    -DCMAKE_BUILD_TYPE=${RELEASE} \
	-DCMAKE_VERBOSE_MAKEFILE=ON \
    \
    -DNANOVDB_BUILD_TOOLS=ON \
    -DNANOVDB_BUILD_EXAMPLES=ON \
    -DNANOVDB_BUILD_UNITTESTS=ON \
    -DNANOVDB_USE_CUDA=ON \
    -DNANOVDB_USE_OPENGL=ON \
    -DNANOVDB_USE_OPENCL=ON \
    -DNANOVDB_USE_OPTIX=ON \
    -DNANOVDB_USE_OPENVDB=ON \
    -DNANOVDB_USE_TBB=ON \
    -DNANOVDB_USE_ZLIB=ON \
    -DNANOVDB_USE_BLOSC=ON \
    -DNANOVDB_BUILD_TOOLS=ON \
    -DNANOVDB_BUILD_INTERACTIVE_RENDERER=OFF \
	-DNANOVDB_BUILD_DOCS=OFF \
	\
    ${CMAKE_EXTRA} \
    ../..

make -j2
make install
chmod +x ../../__release_core/${NAME}/bin/*
