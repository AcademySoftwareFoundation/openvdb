#!/usr/bin/env bash

set -ex

COMPILER="$1"; shift

if [[ $COMPILER = gcc-* ]] || [[ $COMPILER = g++-* ]]; then
    GCC_TOOLCHAIN=`echo $COMPILER | cut -d"-" -f 2`
    COMPILER=`echo $COMPILER | cut -d"-" -f 1`
    echo "COMPILER=$COMPILER GCC_TOOLCHAIN=$GCC_TOOLCHAIN"

    sudo yum -y install devtoolset-${GCC_TOOLCHAIN}-gcc devtoolset-${GCC_TOOLCHAIN}-gcc-c++ || export RC=true
    source scl_source enable devtoolset-${GCC_TOOLCHAIN} || export RC=true
fi

sudo yum -y install gtest-devel || export RC=true
