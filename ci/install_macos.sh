#!/usr/bin/env bash

set -x

brew update
brew install cmake
brew install ilmbase
brew install openexr
brew install boost
brew install boost-python3 # also installs the dependent python version
brew install cppunit
brew install tbb
brew install zlib
brew install glfw
