#!/usr/bin/env bash

set -ex

NAME="$1"; shift
TOOLCHAIN_FILE="$1"; shift
RELEASE="$1"; shift
CMAKE_EXTRA="$@"

mkdir -p __build_core/${NAME}
pushd __build_core/${NAME}

# print version
#cmake --version

cmake \
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    -DCMAKE_INSTALL_PREFIX="../../__release_core/${NAME}" \
    -DCMAKE_CXX_FLAGS_DebugNoInfo="" \
    -DCMAKE_BUILD_TYPE=${RELEASE} \
	-DCMAKE_VERBOSE_MAKEFILE=ON \
    \
    -DNANOVDB_USE_CUDA=OFF \
    -DNANOVDB_USE_TBB=ON \
    -DNANOVDB_USE_OPENCL=ON \
    -DNANOVDB_USE_BLOSC=ON \
    -DNANOVDB_USE_ZLIB=ON \
    -DNANOVDB_USE_OPTIX=OFF \
	\
	-DNANOVDB_BUILD_EXAMPLES=ON \
    -DNANOVDB_BUILD_BENCHMARK=OFF \
    -DNANOVDB_BUILD_UNITTESTS=ON \
    -DNANOVDB_BUILD_TOOLS=ON \
    -DNANOVDB_USE_GLFW=ON \
	-DNANOVDB_BUILD_DOCS=OFF \
	\
    ${CMAKE_EXTRA} \
    ../..

make -j2
make install
if [ -d ../../__release_core/${NAME}/bin ]; then
    chmod +x ../../__release_core/${NAME}/bin/*
fi
popd