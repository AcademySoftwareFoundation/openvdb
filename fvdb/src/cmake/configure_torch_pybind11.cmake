# Configure PyTorch pybind11 compatibility settings. This module sets up the
# necessary compiler flags and definitions to ensure that pybind11 modules
# are built with the same ABI settings as PyTorch. This emulates the behavior
# in torch's cpp_extension.py used by setuptools-based extensions.
#
# It should be included after get_pybind11.cmake since it relies on pybind11
# being already configured.

# Function to configure a pybind11 target with PyTorch compatibility settings
# This ensures that the target is built with the same ABI settings as PyTorch,
# which is critical for avoiding runtime errors when the extension is loaded.
function(configure_torch_pybind11 target_name)
  # Parse arguments
  set(options CUDA)
  set(oneValueArgs "")
  set(multiValueArgs EXTRA_COMPILE_DEFINITIONS EXTRA_COMPILE_OPTIONS)

  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Target ${target_name} does not exist")
  endif()

  # Get PyTorch's C++11 ABI flag (this is critical for compatibility)
  # PyTorch typically either has _GLIBCXX_USE_CXX11_ABI=0 or _GLIBCXX_USE_CXX11_ABI=1
  # NOTE: disabled because it seems to be redundant with TORCH_CXX_FLAGS
  # execute_process(
  #   COMMAND ${Python3_EXECUTABLE} -c "import sys; sys.path.append('${Python3_SITELIB}'); import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  #   OUTPUT_VARIABLE TORCH_CXX11_ABI
  #   OUTPUT_STRIP_TRAILING_WHITESPACE
  # )

  # Extract PyTorch's pybind11 ABI constants
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sys; sys.path.append('${Python3_SITELIB}'); import torch; print(f'COMPILER_TYPE:{torch._C._PYBIND11_COMPILER_TYPE if hasattr(torch._C, \"_PYBIND11_COMPILER_TYPE\") else \"\"};STDLIB:{torch._C._PYBIND11_STDLIB if hasattr(torch._C, \"_PYBIND11_STDLIB\") else \"\"};BUILD_ABI:{torch._C._PYBIND11_BUILD_ABI if hasattr(torch._C, \"_PYBIND11_BUILD_ABI\") else \"\"}')"
    OUTPUT_VARIABLE TORCH_PYBIND11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  message(STATUS "TORCH_PYBIND11_ABI: ${TORCH_PYBIND11_ABI}")

  # Parse the ABI constants
  foreach(abi_type COMPILER_TYPE STDLIB BUILD_ABI)
    string(REGEX MATCH "${abi_type}:([^;]*)" _ "${TORCH_PYBIND11_ABI}")
    set(PYBIND11_${abi_type} "${CMAKE_MATCH_1}")
  endforeach()

  # Log the detected settings
  message(STATUS "PyTorch C++11 ABI flag: _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
  message(STATUS "PyTorch pybind11 compiler type: ${PYBIND11_COMPILER_TYPE}")
  message(STATUS "PyTorch pybind11 stdlib: ${PYBIND11_STDLIB}")
  message(STATUS "PyTorch pybind11 build ABI: ${PYBIND11_BUILD_ABI}")

  # Set all the necessary compile definitions for the target
  target_compile_definitions(${target_name} PRIVATE
    # Set the C++11 ABI flag to match PyTorch
    #_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}

    # Standard PyTorch extension defines
    TORCH_API_INCLUDE_EXTENSION_H

    # Add ABI constants if they're not empty - ensure they are properly quoted in a single string
    $<$<NOT:$<STREQUAL:${PYBIND11_COMPILER_TYPE},>>:PYBIND11_COMPILER_TYPE=\"${PYBIND11_COMPILER_TYPE}\">
    $<$<NOT:$<STREQUAL:${PYBIND11_STDLIB},>>:PYBIND11_STDLIB=\"${PYBIND11_STDLIB}\">
    $<$<NOT:$<STREQUAL:${PYBIND11_BUILD_ABI},>>:PYBIND11_BUILD_ABI=\"${PYBIND11_BUILD_ABI}\">

    # User-provided extra compile definitions
    ${ARG_EXTRA_COMPILE_DEFINITIONS}
  )

  # Add CUDA-specific definitions if CUDA is enabled
  if(ARG_CUDA)
    target_compile_definitions(${target_name} PRIVATE
      USE_CUDA
      TORCH_EXTENSION_NAME=${target_name}
    )

    # Check if we can add CUDA arch flags
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import sys; sys.path.append('${Python3_SITELIB}'); import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.CUDA_HOME is not None and hasattr(torch.utils.cpp_extension, 'COMMON_NVCC_FLAGS') and torch.utils.cpp_extension.COMMON_NVCC_FLAGS or []))"
      OUTPUT_VARIABLE TORCH_COMMON_NVCC_FLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Add CUDA flags
    if(TORCH_COMMON_NVCC_FLAGS)
      string(REPLACE ";" " " NVCC_FLAGS_STR "${TORCH_COMMON_NVCC_FLAGS}")
      message(STATUS "Adding CUDA compiler flags from PyTorch: ${NVCC_FLAGS_STR}")

      target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${TORCH_COMMON_NVCC_FLAGS}>
      )
    endif()
  endif()

  # Finally, log that we've successfully configured the target
  message(STATUS "Successfully configured ${target_name} with PyTorch pybind11 settings")
endfunction()
