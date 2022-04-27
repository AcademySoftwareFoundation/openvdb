# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CXX and CMAKE_CXX options and definitions for OpenVDB build types

#]=======================================================================]

cmake_minimum_required(VERSION 3.15)

###############################################################################

# General CMake and CXX settings

if(FUTURE_MINIMUM_CMAKE_VERSION)
  if(${CMAKE_VERSION} VERSION_LESS ${FUTURE_MINIMUM_CMAKE_VERSION})
    message(DEPRECATION "Support for CMake versions < ${FUTURE_MINIMUM_CMAKE_VERSION} "
      "is deprecated and will be removed.")
  endif()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${MINIMUM_CXX_STANDARD} CACHE STRING
    "The C++ standard whose features are requested to build OpenVDB components." FORCE)
elseif(CMAKE_CXX_STANDARD LESS ${MINIMUM_CXX_STANDARD})
  message(FATAL_ERROR "Provided C++ Standard is less than the supported minimum."
    "Required is at least \"${MINIMUM_CXX_STANDARD}\" (found ${CMAKE_CXX_STANDARD})")
endif()
if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_CXX_STANDARD)
  if(CMAKE_CXX_STANDARD LESS ${FUTURE_MINIMUM_CXX_STANDARD})
    message(DEPRECATION "C++ < 17 is deprecated and will be removed.")
  endif()
endif()

# Configure MS Runtime

if(WIN32 AND CMAKE_MSVC_RUNTIME_LIBRARY)
  message(STATUS "CMAKE_MSVC_RUNTIME_LIBRARY set to target ${CMAKE_MSVC_RUNTIME_LIBRARY}")

  # Configure Boost library varient on Windows
  if(NOT Boost_USE_STATIC_RUNTIME)
    set(Boost_USE_STATIC_RUNTIME OFF)
    if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreaded OR
       CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebug)
      set(Boost_USE_STATIC_RUNTIME ON)
    endif()
  endif()
  if(NOT Boost_USE_DEBUG_RUNTIME)
    set(Boost_USE_DEBUG_RUNTIME OFF)
    if(CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebugDLL OR
       CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL MultiThreadedDebug)
      set(Boost_USE_DEBUG_RUNTIME ON)
    endif()
  endif()
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if(OPENVDB_ENABLE_RPATH)
  # Configure rpath for installation base on the following:
  # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  # @todo make relocatable?
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

# For CMake's find Threads module which brings in pthread - This flag
# forces the compiler -pthread flag vs -lpthread
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

###############################################################################

# Compiler version checks

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_CLANG_VERSION)
    message(FATAL_ERROR "Insufficient clang++ version. Minimum required is "
      "\"${MINIMUM_CLANG_VERSION}\". Found version \"${CMAKE_CXX_COMPILER_VERSION}\""
    )
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_GCC_VERSION)
    message(FATAL_ERROR "Insufficient g++ version. Minimum required is "
      "\"${MINIMUM_GCC_VERSION}\". Found version \"${CMAKE_CXX_COMPILER_VERSION}\""
    )
  endif()
  if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_GCC_VERSION)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS FUTURE_MINIMUM_GCC_VERSION)
      message(DEPRECATION "Support for GCC versions < ${FUTURE_MINIMUM_GCC_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_ICC_VERSION)
    message(FATAL_ERROR "Insufficient ICC version. Minimum required is "
      "\"${MINIMUM_ICC_VERSION}\". Found version \"${CMAKE_CXX_COMPILER_VERSION}\""
    )
  endif()
  if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_ICC_VERSION)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS FUTURE_MINIMUM_ICC_VERSION)
      message(DEPRECATION "Support for ICC versions < ${FUTURE_MINIMUM_ICC_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_MSVC_VERSION)
    message(FATAL_ERROR "Insufficient MSVC version. Minimum required is "
      "\"${MINIMUM_MSVC_VERSION}\". Found version \"${CMAKE_CXX_COMPILER_VERSION}\""
  )
  endif()
  if(OPENVDB_FUTURE_DEPRECATION AND FUTURE_MINIMUM_MSVC_VERSION)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS FUTURE_MINIMUM_MSVC_VERSION)
      message(DEPRECATION "Support for MSVC versions < ${FUTURE_MINIMUM_MSVC_VERSION} "
        "is deprecated and will be removed.")
    endif()
  endif()
else()
  message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
endif()

###############################################################################

# Increase the number of sections that an object file can contain
add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>")
# Excludes APIs such as Cryptography, DDE, RPC, Shell, and Windows Sockets
add_compile_definitions("$<$<CXX_COMPILER_ID:MSVC>:WIN32_LEAN_AND_MEAN>")
# Disable non-secure CRT library function warnings
# https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/
#     compiler-warning-level-3-c4996?view=vs-2019#unsafe-crt-library-functions
add_compile_definitions("$<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>")
# Disable POSIX function name warnings
# https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/
#     compiler-warning-level-3-c4996?view=vs-2019#posix-function-names
add_compile_definitions("$<$<CXX_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_WARNINGS>")
# If a user has explicitly requested a parallel build, configure this during CMake
if(MSVC_MP_THREAD_COUNT AND MSVC_MP_THREAD_COUNT GREATER 1)
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MP${MSVC_MP_THREAD_COUNT}>")
endif()

if(OPENVDB_CXX_STRICT)
  message(STATUS "Configuring CXX warnings")
  # Clang/AppleClang
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Werror>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wall>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wextra>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wconversion>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wno-sign-conversion>")
  # GNU
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Werror>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wextra>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-pedantic>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wcast-align>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wcast-qual>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wconversion>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wdisabled-optimization>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Woverloaded-virtual>")
else()
  # NO OPENVDB_CXX_STRICT, suppress some warnings
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "Suppressing some noisy MSVC CXX warnings, "
      "set OPENVDB_CXX_STRICT=ON to re-enable them.")
  endif()
  # Conversion from int64_t to long
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4244>")
  # It's not possible to use STL types in DLL interfaces in a portable and
  # reliable way so disable this warning
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4251>")
  # Conversion from size_t to uLong
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4267>")
  # Non dll-interface class used as base for dll-interface class
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4275>")
  # Truncation from 'int' to 'bool'
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4305>")
endif()

if(USE_COLORED_OUTPUT)
  message(STATUS "Enabling colored compiler output")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fcolor-diagnostics>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fdiagnostics-color=always>")
endif()

###############################################################################

# Build Types

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
  CACHE STRING [=[Choose the type of build. CMake natively supports the following options: None Debug Release
    RelWithDebInfo MinSizeRel. OpenVDB additionally supports the following sanitizers and tools:
    coverage tsan asan lsan msan ubsan]=]
  FORCE)

if(CMAKE_BUILD_TYPE EQUAL coverage)
  # use .gcno extension instead of .cc.gcno
  # @note This is an undocumented internal cmake var and does not work
  # with multi config generators
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
endif()

# Note that the thread, address and memory sanitizers are incompatible with each other
set(EXTRA_BUILD_TYPES coverage tsan asan lsan msan ubsan)

# Set all build flags to empty (unless they have been provided)

# DebugNoInfo - An internal build type only used by the OpenVDB CI. no optimizations, no symbols, asserts enabled
set(CMAKE_CXX_FLAGS_DebugNoInfo "" CACHE STRING "Flags used by the C++ compiler during DebugNoInfo builds.")

foreach(TYPE ${EXTRA_BUILD_TYPES})
  set(CMAKE_CXX_FLAGS_${U_TYPE} "" CACHE STRING "Flags used by the C++ compiler during ${TYPE} builds.")
  set(CMAKE_SHARED_LINKER_FLAGS_${U_TYPE} "" CACHE STRING "Flags used by the linker during ${TYPE} builds.")
  set(CMAKE_EXE_LINKER_FLAGS_${U_TYPE} "" CACHE STRING "Flags used by the linker during ${TYPE} builds.")
endforeach()

# Init generator options - we use generator expressions to allow builds with both
# clang and GCC. Sanitizers are currently only configured for clang and GCC.

# Coverage
add_compile_options("$<$<AND:$<CONFIG:COVERAGE>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:--coverage>")
add_link_options("$<$<AND:$<CONFIG:COVERAGE>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:--coverage>")

# ThreadSanitizer
add_compile_options("$<$<AND:$<CONFIG:TSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=thread>")
add_compile_options("$<$<AND:$<CONFIG:TSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-g;-O1>")
add_link_options("$<$<AND:$<CONFIG:TSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=thread>")

# AddressSanitize
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=address>")
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fno-omit-frame-pointer;-g;-O1>")
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>>:-fsanitize-address-use-after-scope;-fno-optimize-sibling-calls>")
# -fsanitize-address-use-after-scope added in GCC 7
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<COMPILE_LANG_AND_ID:CXX,GNU>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,7.0.0>>:-fsanitize-address-use-after-scope>")
add_link_options("$<$<AND:$<CONFIG:ASAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=address>")

# LeakSanitizer
add_compile_options("$<$<AND:$<CONFIG:LSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=leak>")
add_compile_options("$<$<AND:$<CONFIG:LSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fno-omit-frame-pointer;-g;-O1>")
add_link_options("$<$<AND:$<CONFIG:LSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=leak>")

# MemorySanitizer
add_compile_options("$<$<AND:$<CONFIG:MSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=memory>")
add_compile_options("$<$<AND:$<CONFIG:MSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fno-omit-frame-pointer;-g;-O2>")
add_compile_options("$<$<AND:$<CONFIG:MSAN>,$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>>:--fno-optimize-sibling-calls;-fsanitize-memory-track-origins=2>")
add_link_options("$<$<AND:$<CONFIG:MSAN>,$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>>:-fsanitize=memory>")

# UndefinedBehaviour
add_compile_options("$<$<AND:$<CONFIG:UBSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=undefined>")
add_link_options("$<$<AND:$<CONFIG:UBSAN>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-fsanitize=undefined>")

# CMAKE_BUILD_TYPE is ignored for multi config generators i.e. MSVS

get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _isMultiConfig)
  message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")
endif()

