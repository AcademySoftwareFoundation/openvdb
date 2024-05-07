# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  Current and future version requirements for all OpenVDB dependencies,
  including compilers, language features and OpenVDB's ABI.

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)

###############################################################################

# Configure minimum version requirements - some are treated specially and fall
# outside of the DISABLE_DEPENDENCY_VERSION_CHECKS catch
set(MINIMUM_CXX_STANDARD 17)

# @note  ABI always enforced so the correct deprecation messages are available.
# OPENVDB_USE_DEPRECATED_ABI_<VERSION> should be used to circumvent this
math(EXPR MINIMUM_OPENVDB_ABI_VERSION "${OpenVDB_MAJOR_VERSION}-2")
math(EXPR FUTURE_OPENVDB_ABI_VERSION "${OpenVDB_MAJOR_VERSION}+1")

# List of deprecated but supported OpenVDB ABIs for this version of OpenVDB
math(EXPR _PREV_ABI "${OpenVDB_MAJOR_VERSION}-1")
set(OPENVDB_DEPRECATED_ABI_LIST ${MINIMUM_OPENVDB_ABI_VERSION} ${_PREV_ABI})
unset(_PREV_ABI)

if(NOT DISABLE_DEPENDENCY_VERSION_CHECKS)
  # @note  Currently tracking CY2022 of the VFX platform where available

  # @Note  Compiler versions are not really a hard and fast rule, you just need
  # a compiler with complete support for our MINIMUM_CXX_STANDARD (currently 17).
  # What's usually important is that the version of libstdc++ and glibc in use
  # matches. Compilers other than GCC should provide options to ensure this
  # targetting e.g. --gcc-toolchain and -fgnuc-version with Clang.
  set(MINIMUM_GCC_VERSION 9.3.1)
  set(MINIMUM_CLANG_VERSION 5.0)
  set(MINIMUM_ICC_VERSION 19)
  set(MINIMUM_MSVC_VERSION 19.28) # 1928 (Visual Studio 2019 Version 16.8 + 16.9)

  # Should be 1.76 for VFX 22, but only version in apt is 1.73
  set(MINIMUM_BOOST_VERSION 1.73)
  set(MINIMUM_PYBIND_VERSION 2.9.1)
  set(MINIMUM_IMATH_VERSION 3.1)
  set(MINIMUM_OPENEXR_VERSION 3.1)
  set(MINIMUM_ZLIB_VERSION 1.2.7)
  set(MINIMUM_TBB_VERSION 2020.3)
  set(MINIMUM_LLVM_VERSION 10.0.0)
  set(MINIMUM_BLOSC_VERSION 1.17.0)
  set(MINIMUM_GLFW_VERSION 3.1)

  set(MINIMUM_PYTHON_VERSION 3.9.1)
  set(MINIMUM_NUMPY_VERSION 1.20.0)

  set(MINIMUM_GOOGLETEST_VERSION 1.10)
  set(MINIMUM_LOG4CPLUS_VERSION 1.1.2)
  set(MINIMUM_HOUDINI_VERSION 19.5)

  # These always promote warnings rather than errors
  set(MINIMUM_MAYA_VERSION 2017)
  set(MINIMUM_DOXYGEN_VERSION 1.8.8)
endif()

# VFX 23 deprecations to transition to MINIMUM_* variables in OpenVDB 12.0.0
# @note  At the time of writing, any variables that are commented out don't
#   have target transitional versions.

set(FUTURE_MINIMUM_GCC_VERSION 11.2.1)
set(FUTURE_MINIMUM_MSVC_VERSION 19.30) # 1930 (Visual Studio 2022)
# set(FUTURE_MINIMUM_ICC_VERSION 19)

# set(FUTURE_MINIMUM_CXX_STANDARD 20)
set(FUTURE_MINIMUM_CMAKE_VERSION 3.20)
# set(FUTURE_MINIMUM_OPENEXR_VERSION 3.1)
set(FUTURE_MINIMUM_BOOST_VERSION 1.80)
set(FUTURE_MINIMUM_GLFW_VERSION 3.3)
set(FUTURE_MINIMUM_LOG4CPLUS_VERSION 2.0)

# set(FUTURE_MINIMUM_BLOSC_VERSION 1.17.0)
# set(FUTURE_MINIMUM_TBB_VERSION 2020.3)
set(FUTURE_MINIMUM_PYTHON_VERSION 3.10)
set(FUTURE_MINIMUM_NUMPY_VERSION 1.23.0)
# set(FUTURE_MINIMUM_HOUDINI_VERSION 20.0)
set(FUTURE_MINIMUM_LLVM_VERSION 13.0.0)
