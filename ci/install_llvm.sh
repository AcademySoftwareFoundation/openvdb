#!/usr/bin/env bash

set -ex

# Download, build and install LLVM from source

LLVM_VERSION=$1
LLVM_ROOT_DIR=$2
if [ -z $LLVM_VERSION ]; then
    echo "No LLVM version provided for LLVM installation"
    exit -1
fi
if [ -z $LLVM_ROOT_DIR ]; then
    echo "No installation directory provided for LLVM installation"
    exit -1
fi

LLVM_MAJOR_VERSION=$(echo $LLVM_VERSION | cut -d. -f1)
LLVM_MINOR_VERSION=$(echo $LLVM_VERSION | cut -d. -f2)
LLVM_PATCH_VERSION=$(echo $LLVM_VERSION | cut -d. -f3)

echo "LLVM Major Version : $LLVM_MAJOR_VERSION"
echo "LLVM Minor Version : $LLVM_MINOR_VERSION"
echo "LLVM Patch Version : $LLVM_PATCH_VERSION"
echo "Downloading LLVM $LLVM_VERSION src..."

# From LLVM >= 10, use github releases
if [ $LLVM_MAJOR_VERSION -ge 10 ]; then
    LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-$LLVM_VERSION.src.tar.xz"
else
    LLVM_URL="https://releases.llvm.org/$LLVM_VERSION/llvm-$LLVM_VERSION.src.tar.xz"
fi

wget -O llvm-$LLVM_VERSION.src.tar.xz $LLVM_URL
tar -xf llvm-$LLVM_VERSION.src.tar.xz
rm -f llvm-$LLVM_VERSION.src.tar.xz
cd llvm-$LLVM_VERSION.src

echo "Building LLVM $LLVM_VERSION -> $LLVM_ROOT_DIR ..."

mkdir -p .build
cd .build
cmake -DCMAKE_INSTALL_PREFIX=$LLVM_ROOT_DIR -DCMAKE_BUILD_TYPE=Release ../
make -j2
make install
