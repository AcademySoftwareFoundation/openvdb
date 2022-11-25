#!/usr/bin/env bash

set -ex

LLVM_CRT="$1"

git clone -b llvmorg-12.0.0 --depth 1 https://github.com/llvm/llvm-project.git llvm
cd llvm

mkdir .build
cd .build

# @note  currently only test static builds with MT
cmake -B . -S ../llvm -A x64 -G "Visual Studio 17 2022" -Thost=x64 \
    -DCMAKE_INSTALL_PREFIX="${HOME}/llvm_install" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_TOOLS=OFF \
    -DLLVM_USE_CRT_RELEASE=${LLVM_CRT}

cmake --build . --config Release --target install

cd ..

du -h .build -d0
du -h ${HOME}/llvm_install -d0

rm -rf .build
