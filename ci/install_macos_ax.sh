#!/usr/bin/env bash

# Download and install deps from homebrew on macos

LLVM_VERSION=$1
if [ -z $LLVM_VERSION ]; then
    echo "No LLVM version provided for LLVM installation"
    exit -1
fi

brew update
brew install ilmbase
brew install openexr
brew install cmake
brew install boost
brew install cppunit
brew install c-blosc
brew install tbb
brew install llvm@$LLVM_VERSION
brew install zlib
