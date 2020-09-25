#!/usr/bin/env bash

set -ex

uname --all

sudo yum -y install opencl-headers ocl-icd
if [ ! -f /usr/lib/libOpenCL.so ]; then
    sudo ln -s /usr/lib64/libOpenCL.so.1 /usr/lib/libOpenCL.so
fi
sudo yum -y install ImageMagick
sudo yum -y install glfw-devel
sudo yum -y install gtk3-devel

g++ --version

BLOSC_VERSION="1.5.0"

git clone https://github.com/Blosc/c-blosc.git
cd c-blosc

if [ "$BLOSC_VERSION" != "latest" ]; then
    git checkout tags/v${BLOSC_VERSION} -b v${BLOSC_VERSION}
fi

mkdir build
cd build
cmake ../.
make -j4
make install
cd ../..


