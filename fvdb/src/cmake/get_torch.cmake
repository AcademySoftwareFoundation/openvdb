# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Include guard to prevent multiple inclusion
if(DEFINED _GET_TORCH_CMAKE_INCLUDED)
  return()
endif()
set(_GET_TORCH_CMAKE_INCLUDED TRUE)

# find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Check that PyTorch package uses the C++11 ABI
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env PYTHONPATH="${Python3_SITELIB}" "${Python3_EXECUTABLE}" -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_IMPORT_RESULT)

if(NOT TORCH_IMPORT_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to import PyTorch. Please ensure PyTorch is installed in the conda environment.")
endif()

if(NOT TORCH_CXX11_ABI)
  message(FATAL_ERROR "PyTorch package does not use the C++11 ABI. "
    "Please install PyTorch with the C++11 ABI (e.g. conda-forge package).")
endif()

# find site-packages directory
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env PYTHONPATH="${Python3_SITELIB}" "${Python3_EXECUTABLE}" -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE TORCH_PACKAGE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# needed to correctly configure Torch with the conda-forge build
if(DEFINED ENV{CONDA_PREFIX})
  set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CONDA_PREFIX}/targets/x86_64-linux")
endif()

find_package(Torch REQUIRED PATHS "${TORCH_PACKAGE_DIR}/share/cmake/Torch")

# Without this we can't find TH/THC headers
set(TORCH_SOURCE_INCLUDE_DIRS ${TORCH_PACKAGE_DIR}/include)

if(NOT TORCH_PYTHON_LIBRARY)
  message(STATUS "Looking for torch_python library...")

  # Create a list of candidate paths
  set(TORCH_PYTHON_LIBRARY_CANDIDATES "${TORCH_PACKAGE_DIR}/lib/libtorch_python.so")

  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND TORCH_PYTHON_LIBRARY_CANDIDATES "$ENV{CONDA_PREFIX}/lib/libtorch_python.so")
  endif()

  # Iterate through candidates until found
  set(TORCH_PYTHON_LIBRARY_FOUND FALSE)

  foreach(CANDIDATE ${TORCH_PYTHON_LIBRARY_CANDIDATES})
    if(EXISTS "${CANDIDATE}")
      set(TORCH_PYTHON_LIBRARY "${CANDIDATE}")
      message(STATUS "Found libtorch_python.so at: ${TORCH_PYTHON_LIBRARY}")
      set(TORCH_PYTHON_LIBRARY_FOUND TRUE)
      break()
    endif()
  endforeach()

  # If not found, report error
  if(NOT TORCH_PYTHON_LIBRARY_FOUND)
    if(DEFINED ENV{CONDA_PREFIX})
      message(FATAL_ERROR "Could not find libtorch_python.so in any of the search locations.")
    else()
      message(FATAL_ERROR "Could not find libtorch_python.so. CONDA_PREFIX was not defined, so only site-packages location was checked.")
    endif()
  endif()
endif()

message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
