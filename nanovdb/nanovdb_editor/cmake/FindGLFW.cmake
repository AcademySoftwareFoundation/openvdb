# FindGLFW.cmake - Locate the GLFW library

# Define variables for include directories and libraries
find_path(GLFW_INCLUDE_DIR
    NAMES GLFW/glfw3.h
    PATH_SUFFIXES GLFW
    PATHS ${CMAKE_SOURCE_DIR}/external/glfw/include
          /usr/include
          /usr/local/include
    DOC "Path to GLFW include directory"
)

find_library(GLFW_LIBRARY
    NAMES glfw glfw3
    PATHS ${CMAKE_SOURCE_DIR}/external/glfw/lib
          /usr/local/lib
          /usr/lib
    DOC "Path to GLFW library"
)

# Check if both include directory and library were found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLFW REQUIRED_VARS GLFW_INCLUDE_DIR GLFW_LIBRARY)

# If found, set additional variables
if(GLFW_FOUND)
    set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
    set(GLFW_LIBRARIES ${GLFW_LIBRARY})

    # Create an imported target for modern CMake usage
    if(NOT TARGET glfw::glfw)
        add_library(glfw::glfw INTERFACE IMPORTED)
        set_target_properties(glfw::glfw PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${GLFW_LIBRARIES}"
        )
    endif()
endif()
