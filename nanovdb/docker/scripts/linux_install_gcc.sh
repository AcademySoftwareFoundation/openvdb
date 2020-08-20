#!/usr/bin/env bash

set -ex

GCC_TOOLCHAIN="$1"; shift

sudo yum -y install gtest-devel || export RC=true
sudo yum -y install devtoolset-${GCC_TOOLCHAIN}-gcc devtoolset-${GCC_TOOLCHAIN}-gcc-c++ || export RC=true
source scl_source enable devtoolset-${GCC_TOOLCHAIN} || export RC=true
