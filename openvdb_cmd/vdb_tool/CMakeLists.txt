cmake_minimum_required(VERSION 3.18)

# CMP0091 allows for MSVC ABI targetting via CMAKE_MSVC_RUNTIME_LIBRARY
# from CMake 3.15 and above. Must come before project().
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

#project(vdb_tool LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, supported options are: Debug Release."
      FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${OPENVDB_CMAKE_PATH})

# Pseudo library to manage dependencies
add_library(vdb_tool_common INTERFACE)

# Optional components
option(BUILD_TEST "Build unit tests" OFF)

option(OPENVDB_TOOL_USE_NANO "Compile with NanoVDB support" OFF)
option(OPENVDB_TOOL_NANO_USE_ZIP "Compile NanoVDB with zip compression support. Requires OPENVDB_TOOL_USE_NANO=ON to have effect" ON)
option(OPENVDB_TOOL_NANO_USE_BLOSC "Compile NanoVDB with Blosc compression support. Requires OPENVDB_TOOL_USE_NANO=ON to have effect" ON)
option(OPENVDB_TOOL_USE_PNG "Compile with PNG support" OFF)
option(OPENVDB_TOOL_USE_EXR "Compile with EXR support" OFF)
option(OPENVDB_TOOL_USE_JPG "Compile with JPG support" OFF)
option(OPENVDB_TOOL_USE_ABC "Compile with Alembic support" OFF)
option(OPENVDB_TOOL_USE_ALL "Compile with all optional components" OFF)
if(OPENVDB_TOOL_USE_ALL)
  set(OPENVDB_TOOL_USE_NANO ON)
  set(OPENVDB_TOOL_USE_PNG ON)
  set(OPENVDB_TOOL_USE_EXR ON)
  set(OPENVDB_TOOL_USE_JPG ON)
  set(OPENVDB_TOOL_USE_ABC ON)
endif()

if(OPENVDB_TOOL_USE_NANO)
  target_compile_definitions(vdb_tool_common INTERFACE "VDB_TOOL_USE_NANO")
  if(OPENVDB_TOOL_NANO_USE_ZIP)
    target_compile_definitions(vdb_tool_common INTERFACE "NANOVDB_USE_ZIP")
    find_package(ZLIB REQUIRED)
    target_link_libraries(vdb_tool_common INTERFACE ZLIB::ZLIB)
  endif()
  if(OPENVDB_TOOL_NANO_USE_BLOSC)
    target_compile_definitions(vdb_tool_common INTERFACE "NANOVDB_USE_BLOSC")
    find_package(Blosc REQUIRED)
    target_link_libraries(vdb_tool_common INTERFACE blosc)
  endif()
  #target_include_directories(vdb_tool_common INTERFACE ${PROJECT_SOURCE_DIR}/../nanovdb/)
  if(NOT OPENVDB_BUILD_NANOVDB)
    find_package(OpenVDB COMPONENTS nanovdb)
    if(NOT OpenVDB_nanovdb_FOUND OR NOT OpenVDB_FOUND)
        message(FATAL_ERROR
        " Couldn't find NanoVDB\n"
        " Either set OPENVDB_CMAKE_PATH to <OpenVDB install path>/lib/cmake/OpenVDB"
        " or please pass -DUSE_NANOVDB=ON as a cmake argument.")
    endif()
    set(NANOVDB_LIB OpenVDB::nanovdb)
    target_include_directories(vdb_tool_common INTERFACE ${OPENVDB_nanovdb_INCLUDE_DIR})
  else()
    set(NANOVDB_LIB nanovdb)
  endif()
  target_link_libraries(vdb_tool_common INTERFACE ${NANOVDB_LIB})
endif()

if(OPENVDB_TOOL_USE_PNG)
  target_compile_definitions(vdb_tool_common INTERFACE "VDB_TOOL_USE_PNG")
  if(WIN32)
    find_package(libpng CONFIG REQUIRED)
  else()
    find_package(PNG REQUIRED)
  endif()
  target_link_libraries(vdb_tool_common INTERFACE png)
endif()

if(OPENVDB_TOOL_USE_JPG)
  target_compile_definitions(vdb_tool_common INTERFACE "VDB_TOOL_USE_JPG")
  find_package(JPEG REQUIRED)
  target_link_libraries(vdb_tool_common INTERFACE ${JPEG_LIBRARIES})
  target_include_directories(vdb_tool_common INTERFACE ${JPEG_INCLUDE_DIR})
endif()

if(OPENVDB_TOOL_USE_EXR)
  target_compile_definitions(vdb_tool_common INTERFACE "VDB_TOOL_USE_EXR")
  find_package(OpenEXR REQUIRED)
  target_link_libraries(vdb_tool_common INTERFACE OpenEXR::IlmImf)
endif()

if(OPENVDB_TOOL_USE_ABC)
  target_compile_definitions(vdb_tool_common INTERFACE "VDB_TOOL_USE_ABC")
  find_package(Alembic CONFIG REQUIRED)
  target_link_libraries(vdb_tool_common INTERFACE Alembic::Alembic)
endif()

if(WIN32 AND (OPENVDB_TOOL_USE_ALL OR (OPENVDB_TOOL_USE_ABC AND OPENVDB_TOOL_USE_EXR)))
  message(WARNING
    " The OpenEXR and Alembic VCPKG packages are using conflicting Imath versions.\n"
    " Disable one, if you encounter unresolved external symbols")
endif()

# Compiler flags

# GCC flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(vdb_tool_common INTERFACE -Wno-invalid-offsetof -pthread -lpthread)
  target_compile_options(vdb_tool_common INTERFACE "$<$<CONFIG:DEBUG>:-O1>")
endif()


# MSVC flags
# Increase the number of sections that an object file can contain
target_compile_options(vdb_tool_common INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>")
# Excludes APIs such as Cryptography, DDE, RPC, Shell, and Windows Sockets
target_compile_definitions(vdb_tool_common INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:WIN32_LEAN_AND_MEAN>")
# Disable non-secure CRT library function warnings
# https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/
#     compiler-warning-level-3-c4996?view=vs-2019#unsafe-crt-library-functions
target_compile_definitions(vdb_tool_common INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>")
# Disable POSIX function name warnings
# https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/
#     compiler-warning-level-3-c4996?view=vs-2019#posix-function-names
target_compile_definitions(vdb_tool_common INTERFACE "$<$<CXX_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_WARNINGS>")

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(vdb_tool_common INTERFACE "$<$<CONFIG:RELEASE>:/Oi>")

  message(STATUS "Suppressing some noisy MSVC CXX warnings.")
endif()
# Conversion from int64_t to long
target_compile_options(vdb_tool_common INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4244>")
# It's not possible to use STL types in DLL interfaces in a portable and
# reliable way so disable this warning
target_compile_options(vdb_tool_common INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4251>")
# Conversion from size_t to uLong
target_compile_options(vdb_tool_common INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4267>")
# Non dll-interface class used as base for dll-interface class
#target_compile_options(vdb_tool_common INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4275>")
# Truncation from 'int' to 'bool'
#target_compile_options(vdb_tool_common INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4305>")

if(OPENVDB_BUILD_CORE)
    target_link_libraries(vdb_tool_common INTERFACE ${OPENVDB_BINARIES_DEPENDENT_LIBS})
else()
    find_package(OpenVDB)
    if(NOT OpenVDB_FOUND)
        message(FATAL_ERROR
        " Couldn't find OpenVDB\n"
        " Set OPENVDB_CMAKE_PATH to <OpenVDB install path>/lib/cmake/OpenVDB")
    endif()
    target_link_libraries(vdb_tool_common INTERFACE TBB::tbb OpenVDB::openvdb)
endif()

target_include_directories(vdb_tool_common INTERFACE "${Boost_INCLUDE_DIRS}" "${OpenVDB_INCLUDE_DIRS}" "include")

# vdb_tool
add_executable(vdb_tool src/main.cpp)
target_include_directories(vdb_tool PRIVATE vdb_tool_common)
target_link_libraries(vdb_tool PRIVATE vdb_tool_common)
install(TARGETS vdb_tool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


# unit test
if(BUILD_TEST)
  find_package(GTest CONFIG REQUIRED)

  add_executable(vdb_tool_test src/unittest.cpp)
  target_include_directories(vdb_tool_test PRIVATE vdb_tool_common)
  target_link_libraries(vdb_tool_test PRIVATE vdb_tool_common GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main)
endif()
