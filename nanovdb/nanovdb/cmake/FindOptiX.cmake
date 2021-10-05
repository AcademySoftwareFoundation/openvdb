# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0

# The distribution contains only 64 bit libraries.  Error when we have been mis-configured.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  if(WIN32)
    message(SEND_ERROR "Make sure when selecting the generator, you select one with Win64 or x64.")
  endif()
  message(FATAL_ERROR "OptiX only supports builds configured for 64 bits.")
endif()

# search path based on the bit-ness of the build.  (i.e. 64: bin64, lib64; 32:
# bin, lib).  Note that on Mac, the OptiX library is a universal binary, so we
# only need to look in lib and not lib64 for 64 bit builds.
if(NOT APPLE)
  set(bit_dest "64")
else()
  set(bit_dest "")
endif()

unset(OptiX_INSTALL_DIR CACHE)

find_path(OptiX_INSTALL_DIR
  NAMES include/optix.h
  PATHS 
  "${OptiX_ROOT}"
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
  NO_DEFAULT_PATH
  )



if (OptiX_INSTALL_DIR)
  set (OptiX_INCLUDE_DIR "${OptiX_INSTALL_DIR}/include")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
  FOUND_VAR OptiX_FOUND
  REQUIRED_VARS OptiX_INCLUDE_DIR
  )
