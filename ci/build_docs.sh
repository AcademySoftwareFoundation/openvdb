#!/usr/bin/env bash

set -x

mkdir build
cd build
cmake \
    -DOPENVDB_BUILD_UNITTESTS=OFF \
    -DOPENVDB_BUILD_BINARIES=OFF \
    -DOPENVDB_BUILD_CORE=OFF \
    -DOPENVDB_BUILD_DOCS=ON \
    ..
make 2>&1 | tee ./doxygen.log
grep "warning:" ./doxygen.log
test $? -eq 1
