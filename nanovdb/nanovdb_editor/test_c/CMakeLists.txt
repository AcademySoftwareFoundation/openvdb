cmake_minimum_required(VERSION 3.15)
project(NanoVDBEditorTest C)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(TEST_COMPILER "Test Compiler" ON)
option(TEST_COMPUTE "Test Compute" ON)
option(TEST_EDITOR "Test Editor" ON)

# Add executable
add_executable(pnanovdbeditortestcapp
    main.c
)

foreach(TEST_OPTION IN ITEMS TEST_COMPILER TEST_COMPUTE TEST_EDITOR)
    if(${TEST_OPTION})
        target_compile_definitions(pnanovdbeditortestcapp PRIVATE ${TEST_OPTION})
    endif()
endforeach()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT pnanovdbeditortestcapp)

# Include directories
target_include_directories(pnanovdbeditortestcapp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Set the library and shader directories
if(WIN32)
    set(NANOVDB_EDITOR_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/$<CONFIG>)
    set(NANOVDB_EDITOR_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/shaders)
else()
    set(NANOVDB_EDITOR_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/${CMAKE_BUILD_TYPE}/lib)
    set(NANOVDB_EDITOR_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build/${CMAKE_BUILD_TYPE}/shaders)
endif()

file(MAKE_DIRECTORY "${NANOVDB_EDITOR_SHADERS_DIR}/_generated")

# Copy shaders to build directory
if (NOT EXISTS "${PROJECT_BINARY_DIR}/shaders")
    add_custom_command(TARGET pnanovdbeditortestcapp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${NANOVDB_EDITOR_SHADERS_DIR}"
        "${PROJECT_BINARY_DIR}/shaders"
    )
endif()

# Copy libraries to build directory
if(WIN32)
    add_custom_command(TARGET pnanovdbeditortestcapp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${NANOVDB_EDITOR_LIB_DIR}"
        "${PROJECT_BINARY_DIR}/$<CONFIG>"
    )
else()
    add_custom_command(TARGET pnanovdbeditortestcapp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${NANOVDB_EDITOR_LIB_DIR}"
        "${PROJECT_BINARY_DIR}/lib"
    )
endif()

set_target_properties(pnanovdbeditortestcapp PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
