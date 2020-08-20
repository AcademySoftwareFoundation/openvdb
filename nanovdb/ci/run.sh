#!/usr/bin/env bash

./ci/build_core_toolchain.sh ./ci/wine-toolchain.cmake msvc16 Release -DCMAKE_VERBOSE_MAKEFILE=OFF -DNANOVDB_USE_OPENVDB=OFF -DNANOVDB_USE_TBB=OFF -DNANOVDB_USE_ZLIB=OFF -DNANOVDB_USE_BLOSC=OFF -DNANOVDB_BUILD_TOOLS=ON -DNANOVDB_BUILD_UNITTESTS=OFF -DNANOVDB_USE_OPTIX=OFF -DNANOVDB_USE_CUDA=OFF

./ci/build_core.sh clang++ clang Release -DCMAKE_VERBOSE_MAKEFILE=OFF -DNANOVDB_USE_OPTIX=OFF -DNANOVDB_USE_CUDA=OFF
./ci/test_core.sh clang
./ci/test_render.sh clang

./ci/build_core.sh g++ gcc8 Release -DCMAKE_VERBOSE_MAKEFILE=OFF -DNANOVDB_USE_OPTIX=OFF -DNANOVDB_USE_CUDA=OFF
./ci/test_core.sh gcc8
./ci/test_render.sh gcc8

