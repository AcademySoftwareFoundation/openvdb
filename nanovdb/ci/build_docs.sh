#!/usr/bin/env bash

set -x

mkdir build-docs
cd build-docs
cmake \
    -DNANOVDB_BUILD_UNITTESTS=OFF \
    -DNANOVDB_BUILD_EXAMPLES=OFF \
	-DNANOVDB_BUILD_BENCHMARK=OFF \
    -DNANOVDB_BUILD_TOOLS=OFF \
    -DNANOVDB_BUILD_DOCS=ON \
    -DNANOVDB_USE_OPENVDB=OFF \
    -DNANOVDB_USE_OPENCL=OFF \
	-DNANOVDB_USE_CUDA=OFF \
    -DNANOVDB_USE_TBB=OFF \
    -DNANOVDB_USE_ZLIB=OFF \
    -DNANOVDB_USE_BLOSC=OFF \
	-DNANOVDB_USE_OPTIX=OFF \
    ..
make 2>&1 | tee ./doxygen.log
grep "warning:" ./doxygen.log
test $? -eq 1
