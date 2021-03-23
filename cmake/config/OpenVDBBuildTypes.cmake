
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

# Requires add_link_options from 3.13
if(${CMAKE_VERSION} VERSION_LESS 3.13)
  if(${CMAKE_BUILD_TYPE} IN_LIST EXTRA_BUILD_TYPES)
    message(FATAL_ERROR "Build type ${CMAKE_BUILD_TYPE} requires CMake version 3.13 or higher.")
  endif()
  return()
endif()
cmake_minimum_required(VERSION 3.13)

foreach(TYPE ${EXTRA_BUILD_TYPES})
  set(CMAKE_CXX_FLAGS_${U_TYPE} "" CACHE STRING "Flags used by the C++ compiler during ${TYPE} builds.")
  set(CMAKE_SHARED_LINKER_FLAGS_${U_TYPE} "" CACHE STRING "Flags used by the linker during ${TYPE} builds.")
  set(CMAKE_EXE_LINKER_FLAGS_${U_TYPE} "" CACHE STRING "Flags used by the linker during ${TYPE} builds.")
endforeach()

# Init generator options - we use generator expressions to allow builds with both
# clang and GCC. Sanitizers are currently only configured for clang and GCC.
# @todo from CMake 3.15, switch to comma list of CXX_COMPILER_ID.

# Coverage
add_compile_options("$<$<CONFIG:COVERAGE>:--coverage>")
add_link_options("$<$<CONFIG:COVERAGE>:--coverage>")

# ThreadSanitizer
add_compile_options("$<$<AND:$<CONFIG:TSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=thread>")
add_compile_options("$<$<AND:$<CONFIG:TSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-g;-O1>")
add_link_options("$<$<AND:$<CONFIG:TSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=thread>")

# AddressSanitize
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=address>")
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fno-omit-frame-pointer;-g;-O1>")
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize-address-use-after-scope;-fno-optimize-sibling-calls>")
# -fsanitize-address-use-after-scope added in GCC 7
add_compile_options("$<$<AND:$<CONFIG:ASAN>,$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,7.0.0>>:-fsanitize-address-use-after-scope>")
add_link_options("$<$<AND:$<CONFIG:ASAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=address>")

# LeakSanitizer
add_compile_options("$<$<AND:$<CONFIG:LSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=leak>")
add_compile_options("$<$<AND:$<CONFIG:LSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fno-omit-frame-pointer;-g;-O1>")
add_link_options("$<$<AND:$<CONFIG:LSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=leak>")

# MemorySanitizer
add_compile_options("$<$<AND:$<CONFIG:MSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=memory>")
add_compile_options("$<$<AND:$<CONFIG:MSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fno-omit-frame-pointer;-g;-O2>")
add_compile_options("$<$<AND:$<CONFIG:MSAN>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:--fno-optimize-sibling-calls;-fsanitize-memory-track-origins=2>")
add_link_options("$<$<AND:$<CONFIG:MSAN>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=memory>")

# UndefinedBehaviour
add_compile_options("$<$<AND:$<CONFIG:UBSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=undefined>")
add_link_options("$<$<AND:$<CONFIG:UBSAN>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-fsanitize=undefined>")
