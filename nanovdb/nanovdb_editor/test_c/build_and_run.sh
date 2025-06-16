#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Create build directory if it doesn't exist
mkdir -p build

# Clean shader cache
rm -rf build/shaders/_generated/*

# Navigate to build directory
cd build

# Configure with CMake
echo "Configuring project with CMake..."
cmake -DTEST_EDITOR=ON -DTEST_COMPILER=ON -DTEST_COMPUTE=ON ..

# Build the project
echo "Building project..."
cmake --build . --config Release

# Run the executable
echo "Running test application..."
if [ "$(uname)" == "Darwin" ] || [ "$(uname)" == "Linux" ]; then
    # Mac or Linux
    if [ -f "./pnanovdbeditortestcapp" ]; then
        ./pnanovdbeditortestcapp
    else
        echo "Error: Executable not found!"
    fi
else
    # Windows
    if [ -f "./Release/pnanovdbeditortestcapp.exe" ]; then
        "./Release/pnanovdbeditortestcapp.exe"
    else
        echo "Error: Executable not found!"
    fi
fi

# Return to original directory
cd ..
