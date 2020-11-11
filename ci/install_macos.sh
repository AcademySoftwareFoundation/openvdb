#!/usr/bin/env bash

set -x

brew update
brew install cmake
brew install ilmbase
brew install openexr
brew install boost
brew install boost-python3 # also installs the dependent python version
brew install gtest
brew install tbb
brew install zlib
brew install glfw
brew install jq # for trivial parsing of brew json

# Alias python version installed by boost-python3 to path
py_version=$(brew info boost-python3 --json | \
    jq -cr '.[].dependencies[] | select(. | startswith("python"))')
echo "Using python $py_version"
# export for subsequent action steps (note, not exported for this env)
echo "Python_ROOT_DIR=/usr/local/opt/$py_version" >> $GITHUB_ENV
echo "/usr/local/opt/$py_version/bin" >> $GITHUB_PATH

