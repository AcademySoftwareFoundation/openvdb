
cmake_minimum_required(VERSION 3.15)

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
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
  CACHE STRING [=[Choose the type of build. CMake natively supports the following options: None Debug Release
    RelWithDebInfo MinSizeRel. OpenVDB additionally supports the following sanitizers and tools:
    coverage tsan asan lsan msan ubsan]=]
  FORCE)

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
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,7.0.0>>:-fsanitize-address-use-after-scope>")
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

# Intialize extra build type targets where possible

if(NOT TARGET gcov_html)
  find_program(GCOVR_PATH gcovr)
  if(NOT GCOVR_PATH AND CMAKE_BUILD_TYPE STREQUAL "coverage")
    message(WARNING "Unable to initialize gcovr target. coverage build types will still generate gcno files.")
  else()
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
