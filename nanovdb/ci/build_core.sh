#!/usr/bin/env bash

set -ex

NAME="$1"; shift
COMPILER="$1"; shift
CUDA_VER="$1"; shift
RELEASE="$1"; shift
CMAKE_EXTRA="$@"

mkdir -p __build_core/${NAME}
cd __build_core/${NAME}

# print version
cmake --version

if [[ $COMPILER = gcc-* ]] || [[ $COMPILER = g++-* ]]; then
    GCC_TOOLCHAIN=`echo $COMPILER | cut -d"-" -f 2`
    COMPILER=`echo $COMPILER | cut -d"-" -f 1`
    echo "COMPILER=$COMPILER GCC_TOOLCHAIN=$GCC_TOOLCHAIN"
fi

if [[ ! -z "$GCC_TOOLCHAIN" ]]; then
    source scl_source enable devtoolset-${GCC_TOOLCHAIN} || export RC=true
fi

sudo rm -f /usr/local/cuda
sudo ln -s /usr/local/cuda-${CUDA_VER} /usr/local/cuda

echo "CUDA_VERSION=$CUDA_VER"

if [[ -z "$RELEASE" ]]; then
    RELEASE=Release
fi

if [[ $COMPILER = msvc* ]]; then
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=../../ci/wine-toolchain.cmake \
        -DCMAKE_INSTALL_PREFIX="../../__release_core/${NAME}" \
        -DCMAKE_CXX_FLAGS_DebugNoInfo="" \
        -DCMAKE_BUILD_TYPE=${RELEASE} \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        \
        -DNANOVDB_USE_CUDA=OFF \
        -DNANOVDB_USE_TBB=OFF \
        -DNANOVDB_USE_OPENCL=OFF \
        -DNANOVDB_USE_BLOSC=OFF \
        -DNANOVDB_USE_ZLIB=OFF \
        -DNANOVDB_USE_OPTIX=OFF \
        \
        -DNANOVDB_BUILD_EXAMPLES=OFF \
        -DNANOVDB_BUILD_BENCHMARK=OFF \
        -DNANOVDB_BUILD_UNITTESTS=OFF \
        -DNANOVDB_BUILD_TOOLS=ON \
        -DNANOVDB_USE_GLFW=ON \
        -DNANOVDB_BUILD_DOCS=OFF \
        \
        -DNANOVDB_USE_OPENVDB=OFF \
        \
        ${CMAKE_EXTRA} \
        ../..
else
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
        -DNANOVDB_USE_GLFW=OFF \
        -DNANOVDB_BUILD_DOCS=OFF \
        \
        ${CMAKE_EXTRA} \
        ../..
fi
make -j2
make install
chmod +x ../../__release_core/${NAME}/bin/*
