# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Set global download timeout to 600 seconds (10 minutes)
set(CMAKE_DOWNLOAD_TIMEOUT 600 CACHE STRING "Time allowed for downloads in seconds")

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

find_package(Python3 COMPONENTS Development Interpreter)

message(STATUS "CMAKE_BUILD_PARALLEL_LEVEL: ${CMAKE_BUILD_PARALLEL_LEVEL}")

include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_cpm.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_nvtx.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_torch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/get_pybind11.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/configure_torch_pybind11.cmake)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure libfvdb
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/)

# Source files
set(FVDB_BINDINGS_CPP_FILES
    src/python/Bindings.cpp
    src/python/GaussianSplatBinding.cpp
    src/python/GridBatchBinding.cpp
    src/python/JaggedTensorBinding.cpp)

# Build library
Python3_add_library(_Cpp ${FVDB_BINDINGS_CPP_FILES})
if (DEFINED TORCH_PYBIND11_INCLUDE_DIR)
    target_link_libraries(_Cpp PUBLIC torch_pybind11_headers)
else()
    target_link_libraries(_Cpp PUBLIC pybind11::headers)
endif()
set_target_properties(_Cpp PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET hidden
                                VISIBILITY_INLINES_HIDDEN ON)

target_link_options(_Cpp PRIVATE "-Wl,--no-as-needed")

set_target_properties(_Cpp PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    POSITION_INDEPENDENT_CODE ON
    INTERFACE_POSITION_INDEPENDENT_CODE ON)

target_include_directories(_Cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS})

target_link_libraries(_Cpp PRIVATE
    ${TORCH_PYTHON_LIBRARY}
    ${TORCH_LIBRARIES}
    fvdb)

target_compile_options(_Cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
    "-Wno-unused-variable"
    "-fvisibility=hidden"
    "-fvisibility-inlines-hidden"
    "${TORCH_CXX_FLAGS}"
    >)

target_compile_definitions(_Cpp PRIVATE
    TORCH_EXTENSION_NAME=_Cpp
    TORCH_API_INCLUDE_EXTENSION_H)

# Configure the pybind11 module with PyTorch compatibility settings
configure_torch_pybind11(_Cpp)

install(TARGETS _Cpp fvdb
    LIBRARY
    DESTINATION fvdb
    COMPONENT PythonModule
    RUNTIME
    DESTINATION fvdb
    COMPONENT PythonModule)

# Add at the beginning after project() call:
set(VERSION_PY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/fvdb/version.py")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/fvdb/version.py.in" "${VERSION_PY_PATH}" @ONLY)

# Make sure version.py is included in the installation
install(FILES "${VERSION_PY_PATH}" DESTINATION fvdb)
