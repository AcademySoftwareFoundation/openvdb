# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Check that PyTorch package uses the C++11 ABI
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if (NOT TORCH_CXX11_ABI)
    message(FATAL_ERROR "PyTorch package does not use the C++11 ABI. "
                        "Please install PyTorch with the C++11 ABI (e.g. conda-forge package).")
endif()

# needed to correctly configure Torch with the conda-forge build
set(CUDA_TOOLKIT_ROOT_DIR "${CONDA_ENV_PATH}/targets/x86_64-linux")
find_package(Torch REQUIRED)


