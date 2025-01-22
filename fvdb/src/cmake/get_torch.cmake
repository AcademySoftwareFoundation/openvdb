# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# find Python3 and site-packages path
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "if True:
    from distutils import sysconfig as sc
    print(sc.get_python_lib(prefix='', plat_specific=True))"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set (PYTHON_SITE "${CONDA_ENV_PATH}/${PYTHON_SITE}")

# Check that PyTorch package uses the C++11 ABI
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if (NOT TORCH_CXX11_ABI)
    message(FATAL_ERROR "PyTorch package does not use the C++11 ABI. "
                        "Please install PyTorch with the C++11 ABI (e.g. conda-forge package).")
endif()

# find torch, looking in site-packages
set(Torch_DIR ${PYTHON_SITE}/torch/share/cmake/Torch)
# needed to correctly configure Torch with the conda-forge build
set(CUDA_TOOLKIT_ROOT_DIR "${CONDA_ENV_PATH}/targets/x86_64-linux")
find_package(Torch REQUIRED)


