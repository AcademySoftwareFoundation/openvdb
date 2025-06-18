# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

format_files() {
    find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \) -exec clang-format -i {} +
}

pushd ../benchmarks
format_files
popd

pushd ../tests
format_files
popd

pushd ../fvdb
format_files
popd
