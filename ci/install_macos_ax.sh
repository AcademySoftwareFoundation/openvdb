#!/usr/bin/env bash

# Download and install deps from homebrew on macos

LLVM_VERSION=$1
if [ -z $LLVM_VERSION ]; then
    echo "No LLVM version provided for LLVM installation"
    exit -1
fi

brew update
brew install bash
brew install cmake
brew install boost
brew install cppunit
brew install c-blosc
brew install tbb@2020
brew install llvm@$LLVM_VERSION
brew install zlib

# Export TBB paths which are no longer installed to /usr/local (as v2020 is deprecated)
echo "TBB_ROOT=/usr/local/opt/tbb@2020" >> $GITHUB_ENV
echo "/usr/local/opt/tbb@2020/bin" >> $GITHUB_PATH
