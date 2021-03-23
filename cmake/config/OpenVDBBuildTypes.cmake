# Build Types
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
    CACHE STRING [=[Choose the type of build. CMake natively supports the following options: None Debug Release
        RelWithDebInfo MinSizeRel. OpenVDB additionally supports the following sanitizers and tools:
        coverage tsan asan lsan msan ubsan]=]
    FORCE)

# DebugNoInfo - An internal build type only used by the OpenVDB CI. no optimizations, no symbols, asserts enabled
set(CMAKE_CXX_FLAGS_DebugNoInfo ""
  CACHE STRING "Flags used by the C++ compiler during DebugNoInfo builds." FORCE)

# Coverage
set(CMAKE_CXX_FLAGS_COVERAGE "--coverage"
  CACHE STRING "Flags used by the C++ compiler during code coverage builds." FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_COVERAGE "--coverage"
  CACHE STRING "Flags used by the linker during code coverage builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "--coverage"
  CACHE STRING "Flags used by the linker during code coverage builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage"
  CACHE STRING "Flags used by the linker during code coverage builds." FORCE)

# Note that the thread, address and memory sanitizers are icompatible with each other
set(SANITIZERS tsan asan lsan msan ubsan)

# ThreadSanitizer
set(CMAKE_CXX_FLAGS_TSAN "-fsanitize=thread -g -O1"
  CACHE STRING "Flags used by the C++ compiler during ThreadSanitizer builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_TSAN "-fsanitize=thread"
  CACHE STRING "Flags used by the linker during ThreadSanitizer builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_TSAN "-fsanitize=thread"
  CACHE STRING "Flags used by the linker during ThreadSanitizer builds." FORCE)

# AddressSanitize
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1"
    CACHE STRING "Flags used by the C++ compiler during AddressSanitizer builds." FORCE)
  set(CMAKE_SHARED_LINKER_FLAGS_ASAN "-fsanitize=address"
    CACHE STRING "Flags used by the linker during AddressSanitizer builds." FORCE)
  set(CMAKE_EXE_LINKER_FLAGS_ASAN "-fsanitize=address"
    CACHE STRING "Flags used by the linker during AddressSanitizer builds." FORCE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address -fno-optimize-sibling-calls -fno-omit-frame-pointer -g -O1"
    CACHE STRING "Flags used by the C++ compiler during AddressSanitizer builds." FORCE)
endif()

# LeakSanitizer
set(CMAKE_CXX_FLAGS_LSAN "-fsanitize=leak -fno-omit-frame-pointer -g -O1"
  CACHE STRING "Flags used by the C++ compiler during LeakSanitizer builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_LSAN "-fsanitize=leak"
  CACHE STRING "Flags used by the linker during LeakSanitizer builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_LSAN "-fsanitize=leak"
  CACHE STRING "Flags used by the linker during LeakSanitizer builds." FORCE)

# MemorySanitizer
set(CMAKE_CXX_FLAGS_MSAN "-fsanitize=memory -fno-optimize-sibling-calls -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O2"
  CACHE STRING "Flags used by the C++ compiler during MemorySanitizer builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_MSAN "-fsanitize=memory"
  CACHE STRING "Flags used by the linker during MemorySanitizer builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_MSAN "-fsanitize=memory"
  CACHE STRING "Flags used by the linker during MemorySanitizer builds." FORCE)

# UndefinedBehaviour
set(CMAKE_CXX_FLAGS_UBSAN "-fsanitize=undefined"
  CACHE STRING "Flags used by the C++ compiler during UndefinedBehaviourSanitizer builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_UBSAN "-fsanitize=undefined"
  CACHE STRING "Flags used by the linker during UndefinedBehaviourSanitizer builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_UBSAN "-fsanitize=undefined"
  CACHE STRING "Flags used by the linker during UndefinedBehaviourSanitizer builds." FORCE)
