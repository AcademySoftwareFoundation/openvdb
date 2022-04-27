# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  Current and future version requirements for all OpenVDB dependencies,
  including compilers, language features and OpenVDB's ABI.

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)

###############################################################################

# Configure minimum version requirements - some are treated specially and fall
# outside of the DISABLE_DEPENDENCY_VERSION_CHECKS catch
set(MINIMUM_CXX_STANDARD 14)

# @note  ABI always enforced so the correct deprecation messages are available.
# OPENVDB_USE_DEPRECATED_ABI_<VERSION> should be used to circumvent this
set(MINIMUM_OPENVDB_ABI_VERSION 7)
set(FUTURE_MINIMUM_OPENVDB_ABI_VERSION 9)
set(FUTURE_OPENVDB_ABI_VERSION 10)

if(NOT DISABLE_DEPENDENCY_VERSION_CHECKS)
  # @note  Currently tracking CY2020 of the VFX platform where available
  set(MINIMUM_GCC_VERSION 6.3.1)
  set(MINIMUM_CLANG_VERSION 3.8)
  set(MINIMUM_ICC_VERSION 17)
  set(MINIMUM_MSVC_VERSION 19.10)

  set(MINIMUM_BOOST_VERSION 1.70)
  set(MINIMUM_ILMBASE_VERSION 2.4)
  set(MINIMUM_OPENEXR_VERSION 2.4)
  set(MINIMUM_ZLIB_VERSION 1.2.7)
  set(MINIMUM_TBB_VERSION 2019.0)
  set(MINIMUM_LLVM_VERSION 7.0.0)
  set(MINIMUM_BLOSC_VERSION 1.5.0)

  set(MINIMUM_PYTHON_VERSION 2.7) # @warning should be 3.7.x, but H18.5+ can still be used with 2.7.x
  set(MINIMUM_NUMPY_VERSION 1.17.0)

  set(MINIMUM_GOOGLETEST_VERSION 1.10)
  set(MINIMUM_GLFW_VERSION 3.1)
  set(MINIMUM_LOG4CPLUS_VERSION 1.1.2)
  set(MINIMUM_HOUDINI_VERSION 18.5)

  # These always promote warnings rather than errors
  set(MINIMUM_MAYA_VERSION 2017)
  set(MINIMUM_DOXYGEN_VERSION 1.8.8)
endif()

# VFX 21 deprecations to transition to MINIMUM_* variables in OpenVDB 10.0.0

set(FUTURE_MINIMUM_GCC_VERSION 9.3.1)
set(FUTURE_MINIMUM_ICC_VERSION 19)
# set(FUTURE_MINIMUM_MSVC_VERSION 19.10)

set(FUTURE_MINIMUM_CXX_STANDARD 17)
set(FUTURE_MINIMUM_CMAKE_VERSION 3.18)
#set(FUTURE_MINIMUM_ILMBASE_VERSION 2.4) # Minimum is already 2.4
#set(FUTURE_MINIMUM_OPENEXR_VERSION 2.4) # Minimum is already 2.4
set(FUTURE_MINIMUM_BOOST_VERSION 1.73)
set(FUTURE_MINIMUM_BLOSC_VERSION 1.17.0)
set(FUTURE_MINIMUM_TBB_VERSION 2020.2)
set(FUTURE_MINIMUM_PYTHON_VERSION 3.7)
set(FUTURE_MINIMUM_NUMPY_VERSION 1.19.0)
set(FUTURE_MINIMUM_HOUDINI_VERSION 19.0)
set(FUTURE_MINIMUM_LLVM_VERSION 10.0.0)
