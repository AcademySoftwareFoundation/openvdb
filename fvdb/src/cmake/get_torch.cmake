# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Check that PyTorch package uses the C++11 ABI
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sys; sys.path.append('${Python3_SITELIB}'); import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_IMPORT_RESULT)

if (NOT TORCH_IMPORT_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to import PyTorch. Please ensure PyTorch is installed in the conda environment.")
endif()

if (NOT TORCH_CXX11_ABI)
    message(FATAL_ERROR "PyTorch package does not use the C++11 ABI. "
                        "Please install PyTorch with the C++11 ABI (e.g. conda-forge package).")
endif()

# find site-packages directory
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# needed to correctly configure Torch with the conda-forge build
set(CUDA_TOOLKIT_ROOT_DIR "${CONDA_ENV_PATH}/targets/x86_64-linux")
find_package(Torch REQUIRED)

# Without this we can't find TH/THC headers
set(TORCH_SOURCE_INCLUDE_DIRS ${PYTHON_SITE_PACKAGES}/torch/include)
