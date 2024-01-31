# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CXX and CMAKE_CXX options and definitions for OpenVDB build types

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)

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
# Enable MSVC options that make it behave like other compilers
add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")
add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:throwingNew>")
add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:inline>")
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

if(MSVC_MP_THREAD_COUNT AND MSVC_MP_THREAD_COUNT GREATER 1)
  # If a user has explicitly requested a parallel build, configure this during CMake
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MP${MSVC_MP_THREAD_COUNT}>")

  # If building with multiple threads with MSVC, delay generation of PDB files
  # until the end of compilation to speed up their generation (invoke mspdbsrv once).
  # This assumes /Zi is being used to generate PDBs (CMakes default)
  add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,MSVC>>:/Zf>")
endif()

if(MSVC_COMPRESS_PDB)
  # /Zi enables PDBs - CMake seems to default to creating PDBs as the defacto way
  # of handling debug symbols. As these PDBs can be very large, we attempt to compress
  # them here with various settings. This disable incremental linking and can make
  # stack traces more confusing.
  # https://devblogs.microsoft.com/cppblog/shrink-my-program-database-pdb-file/

  # First, generate comdats to allow the linker to remove unused data
  add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,MSVC>>:/Gy>")
  # Remove unreferenced packaged functions and data from comdats
  add_link_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,MSVC>>:/OPT:REF>")
  # Fold duplicate comdat data (1 iteration)
  add_link_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,MSVC>>:/OPT:ICF>")
  # Generally compress the generated pdbs as they are being created
  add_link_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,MSVC>>:/PDBCOMPRESS>")
endif()

if(OPENVDB_CXX_STRICT)
  message(STATUS "Configuring CXX warnings")
  # Clang/AppleClang
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Werror>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wall>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wextra>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wconversion>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wnon-virtual-dtor>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wover-aligned>")
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wimplicit-fallthrough>")
  add_compile_options("$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,9.3.0>>:-Wimplicit-fallthrough>")
  # Only check global constructors for libraries (we should really check for
  # executables too but gtest relies on these types of constructors for its
  # framework).
  add_compile_options("$<$<AND:$<NOT:$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>>,$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>>:-Wglobal-constructors>")
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
  add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wnon-virtual-dtor>")
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
set(EXTRA_BUILD_TYPES coverage tsan asan lsan msan ubsan abicheck)

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
# --coverage uses -fprofile-arcs -ftest-coverage (compiling) and -lgcov (linking)
# @note consider -fprofile-abs-path from gcc 10
# @todo consider using clang with source analysis: -fprofile-instr-generate -fcoverage-mapping.
#   https://clang.llvm.org/docs/SourceBasedCodeCoverage.html
#   note that clang also works with gcov (--coverage)
# @note Ideally we'd use no optimisations (-O0) with --coverage, but a complete
#   run of all unit tests takes upwards of a day without them. Thread usage also
#   impacts total runtime. -Og implies -O1 but without optimisations "that would
#   otherwise interfere with debugging". This still massively effects branch
#   coverage tracking compared to -O0 so we should look to improve the speed of
#   some of the unit tests and also experiment with clang.
add_compile_options("$<$<AND:$<CONFIG:COVERAGE>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:--coverage;-Og>")
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

# ABI Check. This build type is expected to work with the abi-dumper/abi-compliance-checker
# binaries which expect specific debug information. In particular, for GCC versions >= 11
# we have to explicitly select dwarf versions < 5 as the abi-dumper doesn't support dwarf5
# and will always incorrectly report successful ABI checks
#   https://github.com/lvc/abi-dumper/issues/33
add_compile_options("$<$<CONFIG:ABICHECK>:-gdwarf-4;-g3;-ggdb;-Og>")

# CMAKE_BUILD_TYPE is ignored for multi config generators i.e. MSVS

get_property(_isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _isMultiConfig)
  message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")
endif()
if(OPENVDB_ENABLE_ASSERTS)
  message(STATUS "OpenVDB asserts are ENABLED")
endif()

# Intialize extra build type targets where possible

if(NOT TARGET gcov_html)
  find_program(GCOVR_PATH gcovr)
  if(NOT GCOVR_PATH AND CMAKE_BUILD_TYPE STREQUAL "coverage")
    message(WARNING "Unable to initialize gcovr target. coverage build types will still generate gcno files.")
  elseif(GCOVR_PATH)
    # init gcov commands
    set(GCOVR_HTML_FOLDER_CMD ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/gcov_html)
    set(GCOVR_HTML_CMD
      ${GCOVR_PATH} --html --html-details -r ${PROJECT_SOURCE_DIR} --object-directory=${PROJECT_BINARY_DIR}
      -o gcov_html/index.html
    )

    # Add a custom target which converts .gcda files to a html report using gcovr.
    # Note that this target does NOT run ctest or any binaries - that is left to
    # the implementor of the gcov workflow. Typically, the order of operations
    # would be:
    #  - run CMake with unit tests on
    #  - ctest
    #  - make gcov_html
    add_custom_target(gcov_html
      COMMAND ${GCOVR_HTML_FOLDER_CMD}
      COMMAND ${GCOVR_HTML_CMD}
      BYPRODUCTS ${PROJECT_BINARY_DIR}/gcov_html/index.html  # report directory
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      VERBATIM
      COMMENT "Running gcovr to produce HTML code coverage report."
    )

    # Show info where to find the report
    add_custom_command(TARGET gcov_html POST_BUILD
      COMMAND ;
      COMMENT "Open ./gcov_html/index.html in your browser to view the coverage report."
    )
  endif()
endif()
