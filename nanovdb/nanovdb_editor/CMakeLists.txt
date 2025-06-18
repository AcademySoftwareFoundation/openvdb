# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB Editor

#]=======================================================================]

option(NANOVDB_EDITOR_USE_VCPKG "Use Vcpkg" OFF)
option(NANOVDB_EDITOR_USE_GUI "Use GUI" ON)
option(NANOVDB_EDITOR_CLEAN_SHADERS "Clean shader cache (_generated dir)" OFF)
option(NANOVDB_EDITOR_SLANG_DEBUG_OUTPUT "Enable Slang debug output (compiles shader into Spirv Assembly)" OFF)
option(NANOVDB_EDITOR_ASAN_DEBUG_BUILD "Compile with AddressSanitizer enabled" OFF)
option(NANOVDB_EDITOR_DEBUG_PYTHON "Compile with debug Python" OFF)

set(NANOVDB_EDITOR_SLANG_PROFILE "spirv_1_3" CACHE STRING "Slang shader profile to use")

cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_JOBS auto)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")       # local cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake") # openvdb cmake

include(FetchContent)
include(pnanovdb_cmake_utils)

project(NanoVDBEditor LANGUAGES CXX)

### Directory Setup

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(SHADERS_DIR_NAME shaders)
set(INCLUDES_DIR includes)
set(COMPILED_SHADERS_DIR_NAME _generated)
set(COMPILED_SHADERS_TEMP_DIR_NAME nanovdb_editor/shader_cache)

add_definitions(-DNANOVDB_EDITOR_SHADER_DIR="${SHADERS_DIR_NAME}")
add_definitions(-DNANOVDB_EDITOR_INCLUDES_DIR="${INCLUDES_DIR}")
add_definitions(-DNANOVDB_EDITOR_SHADER_CACHE="${SHADERS_DIR_NAME}/${COMPILED_SHADERS_DIR_NAME}")
add_definitions(-DNANOVDB_EDITOR_TEMP_SHADER_CACHE="${COMPILED_SHADERS_TEMP_DIR_NAME}")
add_definitions(-DNANOVDB_EDITOR_USER_PARAMS_SIZE=64)

# set the shaders directory next to the lib directory
set(SHADERS_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../${SHADERS_DIR_NAME}")
set(COMPILED_SHADERS_DIR "${SHADERS_DIR}/${COMPILED_SHADERS_DIR_NAME}")

if(NANOVDB_EDITOR_CLEAN_SHADERS)
  if(EXISTS "${COMPILED_SHADERS_DIR}")
      file(REMOVE_RECURSE "${COMPILED_SHADERS_DIR}")
      message(STATUS "Removed existing folder for shader cache: ${COMPILED_SHADERS_DIR}")
  endif()
endif()

if(NOT EXISTS "${COMPILED_SHADERS_DIR}")
  file(MAKE_DIRECTORY "${COMPILED_SHADERS_DIR}")
  message(STATUS "Created folder for shader cache: ${COMPILED_SHADERS_DIR}")
endif()

set(DATA_DIR ${CMAKE_SOURCE_DIR}/data/)
file(MAKE_DIRECTORY "${DATA_DIR}")

### Python Module

set(NANOVDB_EDITOR_PYTHON_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pymodule/nanovdb_editor/lib)
set(NANOVDB_EDITOR_PYTHON_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pymodule/nanovdb_editor/shaders)

# set library config for python module
if(NANOVDB_EDITOR_DEBUG_PYTHON)
  set(PYTHON_LIB_CONFIG Debug)
else()
  set(PYTHON_LIB_CONFIG Release)
endif()

if(WIN32)
    set(NANOVDB_EDITOR_LIB_DIR "${PROJECT_BINARY_DIR}/${PYTHON_LIB_CONFIG}")
else()
  if(CMAKE_BUILD_TYPE STREQUAL ${PYTHON_LIB_CONFIG})
    set(NANOVDB_EDITOR_LIB_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  endif()
endif()

if(DEFINED NANOVDB_EDITOR_LIB_DIR)
  pnanovdb_prebuild_link(
    "${NANOVDB_EDITOR_LIB_DIR}"
    "${NANOVDB_EDITOR_PYTHON_LIB_DIR}"
  )
endif()

pnanovdb_prebuild_link(
  "${SHADERS_DIR}"
  "${NANOVDB_EDITOR_PYTHON_SHADERS_DIR}"
)

if(SKBUILD)
  install(TARGETS pnanovdbcompiler pnanovdbcompute pnanovdbeditor
    RUNTIME DESTINATION ${NANOVDB_EDITOR_PYTHON_LIB_DIR}  # For Windows DLLs
    LIBRARY DESTINATION ${NANOVDB_EDITOR_PYTHON_LIB_DIR}  # For Linux/Mac shared libraries
  )
endif()

### Dependencies

# Download CPM.cmake
# Will get Find*.cmake from https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html#find-modules
# Used for: FindXercesC.cmake
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

if(!NANOVDB_ALLOW_FETCHCONTENT)
  message(FATAL_ERROR "Could not fetch dependencies")
endif()

function(check_network_connectivity RESULT)
  if(WIN32)
    execute_process(
      COMMAND ping -n 1 github.com
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET
      ERROR_QUIET
    )
  else()
    execute_process(
      COMMAND ping -c 1 github.com
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET
      ERROR_QUIET
    )
  endif()

  if(EXIT_CODE EQUAL 0)
    set(${RESULT} TRUE PARENT_SCOPE)
  else()
    set(${RESULT} FALSE PARENT_SCOPE)
  endif()
endfunction()

check_network_connectivity(ONLINE)
if(NOT ONLINE)
  set(FETCHCONTENT_FULLY_DISCONNECTED ON)
  message(STATUS "Network connectivity check failed - switching to offline mode")
endif()

if(NANOVDB_EDITOR_USE_VCPKG)
  find_package(glfw3 REQUIRED)
  find_package(Blosc REQUIRED)
  find_package(e57format REQUIRED)
  find_package(boost_regex REQUIRED)  # for ImgUiColorTextEdit
  find_package(ZLIB REQUIRED)
else()
  if(APPLE)
    find_package(GLFW REQUIRED)
  else()
    include(OpenVDBGLFW3Setup)
  endif()
  find_package(Blosc REQUIRED)
  find_package(ZLIB REQUIRED)

  # LibE57Format
  message(STATUS "Fetching libE57Format...")
  FetchContent_Declare(
    libE57Format
    GIT_REPOSITORY https://github.com/asmaloney/libE57Format.git
    GIT_TAG        v3.2.0
  )
  FetchContent_MakeAvailable(libE57Format)
endif()

# Vulkan headers
message(STATUS "Fetching Vulkan headers...")
FetchContent_Declare(
  VulkanHeaders
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG        vulkan-sdk-1.4.309
)
FetchContent_MakeAvailable(VulkanHeaders)

# Slang
message(STATUS "Fetching Slang...")
set(SLANG_RELEASE "2025.6.4")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(SLANG_URL https://github.com/shader-slang/slang/releases/download/v${SLANG_RELEASE}/slang-${SLANG_RELEASE}-windows-x86_64.zip)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(SLANG_URL https://github.com/shader-slang/slang/releases/download/v${SLANG_RELEASE}/slang-${SLANG_RELEASE}-linux-x86_64.zip)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(SLANG_URL https://github.com/shader-slang/slang/releases/download/v${SLANG_RELEASE}/slang-${SLANG_RELEASE}-macos-aarch64.zip)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()
FetchContent_Declare(
  Slang
  URL ${SLANG_URL}
)
FetchContent_MakeAvailable(Slang)

# Imgui Docking
message(STATUS "Fetching Imgui...")
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.91.8-docking
)
FetchContent_MakeAvailable(imgui)
file(GLOB IMGUI_SOURCE_FILES "${imgui_SOURCE_DIR}/imgui*.cpp")

# Json
message(STATUS "Fetching json...")
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# FileWatch
message(STATUS "Fetching FileWatch...")
FetchContent_Declare(
  FileWatch
  GIT_REPOSITORY https://github.com/ThomasMonkman/filewatch.git
  GIT_TAG        master
)
if(NOT FileWatch_POPULATED)
  FetchContent_Populate(FileWatch)  # MakeAvailable imported also the tests
endif()

# Cnpy
message(STATUS "Fetching Cnpy...")
FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(cnpy)

# ImGuiFileDialog
message(STATUS "Fetching ImGuiFileDialog...")
FetchContent_Declare(
  ImGuiFileDialog
  GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
  GIT_TAG        v0.6.7
)
add_compile_definitions(
  IMGUI_INCLUDE="${imgui_SOURCE_DIR}/imgui.h"
  IMGUI_INTERNAL_INCLUDE="${imgui_SOURCE_DIR}/imgui_internal.h"
)
if(NOT ImGuiFileDialog_POPULATED)
  FetchContent_Populate(ImGuiFileDialog)  # Do not build and link the library
  file(GLOB IMGUIFILEDIALOG_SOURCE_FILES "${imguifiledialog_SOURCE_DIR}/*.cpp")
endif()

# ImGuiColorTextEdit
message(STATUS "Fetching ImGuiColorTextEdit...")
FetchContent_Declare(
  ImGuiColorTextEdit
  GIT_REPOSITORY https://github.com/santaclose/ImGuiColorTextEdit.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(ImGuiColorTextEdit)
file(GLOB IMGUICOLORTEXTEDIT_SOURCE_FILES "${imguicolortextedit_SOURCE_DIR}/*.cpp")

# zstr - C++ zlib wrapper
message(STATUS "Fetching zstr...")
FetchContent_Declare(
  zstr
  GIT_REPOSITORY https://github.com/mateidavid/zstr.git
  GIT_TAG        v1.0.7
)
FetchContent_MakeAvailable(zstr)

### PNanoVDB Compiler

file(GLOB COMPILER_SOURCE_FILES "compiler/*.cpp")

set(COMPILER_INCLUDE_DIRS
  ../
  ./compiler
)

add_library(pnanovdbcompiler SHARED
  ${COMPILER_SOURCE_FILES}
)

target_include_directories(pnanovdbcompiler PRIVATE
  ${COMPILER_INCLUDE_DIRS}
  ${slang_SOURCE_DIR}/include
)

# add headers as sources for IDE visibility
file(GLOB COMPILER_HEADER_FILES ${CMAKE_SOURCE_DIR}/compiler/*.h)
target_sources(pnanovdbcompiler PRIVATE ${COMPILER_HEADER_FILES})

target_link_directories(pnanovdbcompiler PRIVATE
  ${slang_SOURCE_DIR}/lib
)

target_link_libraries(pnanovdbcompiler PRIVATE
  slang
  nlohmann_json::nlohmann_json
)

if(APPLE)
  target_link_libraries(pnanovdbcompiler PRIVATE "-framework CoreFoundation")
  target_link_libraries(pnanovdbcompiler PRIVATE "-framework CoreServices")
endif()

if(WIN32)
  add_custom_command(TARGET pnanovdbcompiler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${slang_SOURCE_DIR}/bin/slang.dll "${PROJECT_BINARY_DIR}/$<CONFIG>"
  )
  add_custom_command(TARGET pnanovdbcompiler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${slang_SOURCE_DIR}/bin/slang-glslang.dll "${PROJECT_BINARY_DIR}/$<CONFIG>"
  )
  add_custom_command(TARGET pnanovdbcompiler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${slang_SOURCE_DIR}/bin/slang-llvm.dll "${PROJECT_BINARY_DIR}/$<CONFIG>"
  )
endif()

if(NANOVDB_EDITOR_SLANG_DEBUG_OUTPUT)
  target_compile_definitions(pnanovdbcompiler PRIVATE
    USE_SLANG_DEBUG_OUTPUT
  )
endif()

target_compile_definitions(pnanovdbcompiler PRIVATE
  SLANG_PROFILE=${NANOVDB_EDITOR_SLANG_PROFILE}
)

if(APPLE)
  set_target_properties(pnanovdbcompiler PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path;@loader_path/../_deps/slang-src/lib"
    INSTALL_NAME_DIR "@rpath"
  )
else()
  set_target_properties(pnanovdbcompiler PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN:$ORIGIN/../_deps/slang-src/lib"
  )
endif()

### PNanoVDB Compute

file(GLOB COMPUTE_SOURCE_FILES "compute/*.cpp")
file(GLOB VULKAN_SOURCE_FILES "vulkan/*.cpp")
file(GLOB RASTER_SOURCE_FILES "raster/*.cpp")

set(COMPUTE_INCLUDE_DIRS
  ../
  ./compute
  ./vulkan
  ./raster
)

add_library(pnanovdbcompute SHARED
  ${COMPUTE_SOURCE_FILES}
  ${VULKAN_SOURCE_FILES}
  ${RASTER_SOURCE_FILES}
)

target_include_directories(pnanovdbcompute PRIVATE
  ${COMPUTE_INCLUDE_DIRS}
  ${vulkanheaders_SOURCE_DIR}/include
)

# add headers as sources for IDE visibility
file(GLOB COMPUTE_HEADER_FILES ${CMAKE_SOURCE_DIR}/compute/*.h)
file(GLOB VULKAN_HEADER_FILES ${CMAKE_SOURCE_DIR}/vulkan/*.h)
target_sources(pnanovdbcompute PRIVATE
  ${COMPUTE_HEADER_FILES}
  ${VULKAN_HEADER_FILES}
)

target_link_libraries(pnanovdbcompute PRIVATE
  nlohmann_json::nlohmann_json
  Blosc::blosc
)

target_compile_definitions(pnanovdbcompute PRIVATE
  NOMINMAX
  NANOVDB_USE_BLOSC
)

if(APPLE)
  set_target_properties(pnanovdbcompute PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path"
    INSTALL_NAME_DIR "@rpath"
  )
else()
  set_target_properties(pnanovdbcompute PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN"
  )
endif()

### PNanoVDB FileFormat

file(GLOB FILEFORMAT_SOURCE_FILES "fileformat/*.cpp")

add_library(pnanovdbfileformat SHARED
  ${FILEFORMAT_SOURCE_FILES}
)

# add headers as sources for IDE visibility
file(GLOB FILEFORMAT_HEADER_FILES ${CMAKE_SOURCE_DIR}/fileformat/*.h)
target_sources(pnanovdbfileformat PRIVATE ${FILEFORMAT_HEADER_FILES})

target_include_directories(pnanovdbfileformat PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${cnpy_SOURCE_DIR}
  ${zstr_SOURCE_DIR}/src
)

target_link_libraries(pnanovdbfileformat PRIVATE
  nlohmann_json::nlohmann_json
  cnpy
  E57Format
  ZLIB::ZLIB
)

if(APPLE)
  set_target_properties(pnanovdbfileformat PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path"
    INSTALL_NAME_DIR "@rpath"
  )
else()
  set_target_properties(pnanovdbfileformat PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN"
  )
endif()

### PNanoVDB Editor

file(GLOB EDITOR_SOURCE_FILES "editor/*.cpp")
file(GLOB VULKAN_IMGUI_SOURCE_FILES "imgui/*.cpp")

add_library(pnanovdbeditor SHARED
  ${EDITOR_SOURCE_FILES}
  ${VULKAN_IMGUI_SOURCE_FILES}
  ${IMGUI_SOURCE_FILES}
  ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp           # for std::string text input
  ${IMGUIFILEDIALOG_SOURCE_FILES}
  ${IMGUICOLORTEXTEDIT_SOURCE_FILES}
)

# add headers as sources for IDE visibility
file(GLOB EDITOR_HEADER_FILES ${CMAKE_SOURCE_DIR}/editor/*.h)
target_sources(pnanovdbeditor PRIVATE ${EDITOR_HEADER_FILES})

target_include_directories(pnanovdbeditor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/imgui
  ${imgui_SOURCE_DIR}
  ${GLFW_INCLUDE_DIR}
  ${filewatch_SOURCE_DIR}
  ${imguifiledialog_SOURCE_DIR}
  ${imguicolortextedit_SOURCE_DIR}
  ${vulkanheaders_SOURCE_DIR}/include
)

target_link_libraries(pnanovdbeditor PRIVATE
  nlohmann_json::nlohmann_json
)

if(WIN32)
  target_link_libraries(pnanovdbeditor PRIVATE Boost::regex)

  get_target_property(GLFW3_DLL glfw IMPORTED_LOCATION_RELEASE)
  add_custom_command(TARGET pnanovdbeditor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${GLFW3_DLL} "${PROJECT_BINARY_DIR}/$<CONFIG>"
  )
else()
  find_library(BOOST_REGEX_LIBRARY
    NAMES boost_regex
    HINTS /opt/homebrew/lib
    HINTS /usr/lib
  )
  if (NOT BOOST_REGEX_LIBRARY)
    message(FATAL_ERROR "Boost regex library not found!")
  endif()
  target_link_libraries(pnanovdbeditor PRIVATE ${BOOST_REGEX_LIBRARY})
endif()

if(APPLE)
  target_link_libraries(pnanovdbeditor PRIVATE "-framework CoreFoundation")
  target_link_libraries(pnanovdbeditor PRIVATE "-framework CoreServices")
endif()

target_compile_definitions(pnanovdbeditor PRIVATE
  NOMINMAX
  TEST_NODE2
)

target_compile_definitions(pnanovdbeditor PRIVATE
  COMPILED_SHADERS_DIR=${COMPILED_SHADERS_DIR}
)

if (NANOVDB_EDITOR_USE_GUI)
  target_compile_definitions(pnanovdbeditor PRIVATE
    USE_IMGUI_INSTANCE
  )
endif()

if(ASAN_DEBUG_BUILD)
  message(STATUS "Compiled with ASan")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

  target_link_options(pnanovdbeditor PUBLIC
    -fsanitize=address
  )
  # to supress the shared_ptr error, run pnanovdbeditor with:
  #   ASAN_OPTIONS=detect_odr_violation=0 ./build/Debug/pnanovdbeditor
endif()

if(APPLE)
  set_target_properties(pnanovdbeditor PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path"
    INSTALL_NAME_DIR "@rpath"
  )
else()
  set_target_properties(pnanovdbeditor PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN"
  )
endif()

# prebuild

set(SHADER_NANOVDB_DIR ${PROJECT_BINARY_DIR}/shaders/nanovdb)
file(MAKE_DIRECTORY "${SHADER_NANOVDB_DIR}")

file(COPY "${CMAKE_SOURCE_DIR}/includes"
     DESTINATION "${PROJECT_BINARY_DIR}/shaders")
file(CREATE_LINK "${CMAKE_SOURCE_DIR}/../nanovdb/PNanoVDB.h" "${PROJECT_BINARY_DIR}/shaders/nanovdb/PNanoVDB.h" SYMBOLIC COPY_ON_ERROR)
file(CREATE_LINK "${CMAKE_SOURCE_DIR}/../nanovdb/PNanoVDB2.h" "${PROJECT_BINARY_DIR}/shaders/nanovdb/PNanoVDB2.h" SYMBOLIC COPY_ON_ERROR)

pnanovdb_prebuild_link(
  "${CMAKE_SOURCE_DIR}/../nanovdb/putil"
  "${PROJECT_BINARY_DIR}/shaders/nanovdb/putil")

pnanovdb_prebuild_link(
  "${DATA_DIR}"
  "${PROJECT_BINARY_DIR}/data"
)

pnanovdb_prebuild_link(
  "${CMAKE_SOURCE_DIR}/compute/shaders"
  "${PROJECT_BINARY_DIR}/shaders/compute")

pnanovdb_prebuild_link(
  "${CMAKE_SOURCE_DIR}/imgui/shaders"
  "${PROJECT_BINARY_DIR}/shaders/imgui"
)

pnanovdb_prebuild_link(
  "${CMAKE_SOURCE_DIR}/test/shaders"
  "${PROJECT_BINARY_DIR}/shaders/test"
)

pnanovdb_prebuild_link(
  "${CMAKE_SOURCE_DIR}/editor/shaders"
  "${PROJECT_BINARY_DIR}/shaders/editor"
)

pnanovdb_prebuild_link(
  "${CMAKE_SOURCE_DIR}/raster/shaders"
  "${PROJECT_BINARY_DIR}/shaders/raster")

# ### PNanoVDB Editor App

file(GLOB EDITOR_APP_SOURCE_FILES "app/*.cpp")

add_executable(pnanovdbeditorapp
  ${EDITOR_APP_SOURCE_FILES}
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT pnanovdbeditorapp)

target_include_directories(pnanovdbeditorapp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/app
  ${CMAKE_CURRENT_SOURCE_DIR}/editor
  ${imgui_SOURCE_DIR}
)

target_link_libraries(pnanovdbeditorapp PRIVATE
  Blosc::blosc
)

target_compile_definitions(pnanovdbeditorapp PRIVATE
  NOMINMAX
  TEST_NODE2
)

if(APPLE)
  set_target_properties(pnanovdbeditorapp PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/lib;/usr/local/lib"
    INSTALL_NAME_DIR "@rpath"
  )
else()
  set_target_properties(pnanovdbeditorapp PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/lib"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )
endif()

### PNanoVDB Editor Test App

add_executable(pnanovdbeditortestapp
  test/main.cpp
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT pnanovdbeditortestapp)

target_include_directories(pnanovdbeditortestapp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${slang_SOURCE_DIR}/include
)

target_link_libraries(pnanovdbeditortestapp PRIVATE
)

if(WIN32)
  get_target_property(GLFW3_DLL glfw IMPORTED_LOCATION_RELEASE)
  add_custom_command(TARGET pnanovdbeditortestapp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${GLFW3_DLL} "${PROJECT_BINARY_DIR}/$<CONFIG>"
  )
endif()

target_compile_definitions(pnanovdbeditortestapp PRIVATE
  NOMINMAX
  TEST_NODE2
)

if(APPLE)
  set_target_properties(pnanovdbeditortestapp PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/lib"
    INSTALL_NAME_DIR "@rpath"
  )
else()
  set_target_properties(pnanovdbeditortestapp PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/lib"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )
endif()
