#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -ex

LLVM_VER="$1"
LLVM_CRT="$2"

# Legacy define for older versions of LLVM
LLVM_LEGACY_CRT_DEFINE=""
if [[ "${LLVM_CRT}" == "MultiThreaded" ]]; then
    LLVM_LEGACY_CRT_DEFINE="MT"
elif [[ "${LLVM_CRT}" == "MultiThreadedDebug" ]]; then
    LLVM_LEGACY_CRT_DEFINE="MTd"
elif [[ "${LLVM_CRT}" == "MultiThreadedDLL" ]]; then
    LLVM_LEGACY_CRT_DEFINE="MD"
elif [[ "${LLVM_CRT}" == "MultiThreadedDebugDLL" ]]; then
    LLVM_LEGACY_CRT_DEFINE="MDd"
fi

git clone -b llvmorg-${LLVM_VER} --depth 1 https://github.com/llvm/llvm-project.git llvm
cd llvm

mkdir .build
cd .build

cmake -B . -S ../llvm -A x64 -G "Visual Studio 17 2022" -Thost=x64 \
    -DCMAKE_INSTALL_PREFIX="${HOME}/llvm_install" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_PROJECTS="" \
    -DLLVM_ENABLE_RUNTIMES="" \
    -DLLVM_ENABLE_UNWIND_TABLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_INCLUDE_TOOLS=OFF \
    -DLLVM_INCLUDE_UTILS=OFF \
    -DLLVM_BUILD_BENCHMARKS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_BUILD_DOCS=OFF \
    -DLLVM_BUILD_TOOLS=OFF \
    -DLLVM_BUILD_UTILS=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_PARALLEL_COMPILE_JOBS=4 \
    -DLLVM_PARALLEL_LINK_JOBS=4 \
    -DCMAKE_MSVC_RUNTIME_LIBRARY=${LLVM_CRT} \
    -DLLVM_USE_CRT_RELEASE=${LLVM_LEGACY_CRT_DEFINE}

cmake --build . --parallel 4 --config Release --target install

cd ..

du -h .build -d0
du -h ${HOME}/llvm_install -d0

rm -rf .build
