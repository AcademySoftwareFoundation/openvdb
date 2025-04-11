# Find and use the exact pybind11 version that PyTorch is using
# Use CPM for consistency with other project dependencies

# First make sure Python is found
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get PyTorch's pybind11 version
function(detect_torch_pybind11_version)
    # Try to find pybind11/detail/common.h in TORCH_INCLUDE_DIRS
    foreach(dir ${TORCH_INCLUDE_DIRS})
        if(EXISTS "${dir}/pybind11/detail/common.h")
            set(PYBIND11_HEADER "${dir}/pybind11/detail/common.h")
            set(TORCH_PYBIND11_INCLUDE_DIR "${dir}" PARENT_SCOPE)
            break()
        endif()
    endforeach()

    # If not found, try python site-packages
    if(NOT PYBIND11_HEADER)
        if(EXISTS "${PYTHON_SITE_PACKAGES}/torch/include/pybind11/detail/common.h")
            set(PYBIND11_HEADER "${PYTHON_SITE_PACKAGES}/torch/include/pybind11/detail/common.h")
            set(TORCH_PYBIND11_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/torch/include" PARENT_SCOPE)
        endif()
    endif()

    if(NOT PYBIND11_HEADER)
        message(FATAL_ERROR "Could not find pybind11 headers in PyTorch")
    endif()

    # Extract the version from the header by reading the file content
    file(READ "${PYBIND11_HEADER}" header_content)

    # First try standard version defines
    string(REGEX MATCH "#define PYBIND11_VERSION_MAJOR[ \t]+([0-9]+)" _ "${header_content}")
    set(MAJOR "${CMAKE_MATCH_1}")

    string(REGEX MATCH "#define PYBIND11_VERSION_MINOR[ \t]+([0-9]+)" _ "${header_content}")
    set(MINOR "${CMAKE_MATCH_1}")

    string(REGEX MATCH "#define PYBIND11_VERSION_PATCH[ \t]+([0-9]+)" _ "${header_content}")
    set(PATCH "${CMAKE_MATCH_1}")

    # Check if the regex patterns matched (not if values are non-zero)
    if(DEFINED MAJOR AND DEFINED MINOR AND DEFINED PATCH)
        set(PYBIND11_VERSION_MAJOR "${MAJOR}" PARENT_SCOPE)
        set(PYBIND11_VERSION_MINOR "${MINOR}" PARENT_SCOPE)
        set(PYBIND11_VERSION_PATCH "${PATCH}" PARENT_SCOPE)
        set(PYBIND11_VERSION "${MAJOR}.${MINOR}.${PATCH}" PARENT_SCOPE)
        message(STATUS "Detected PyTorch's pybind11 version: ${MAJOR}.${MINOR}.${PATCH}")
    else()
        message(FATAL_ERROR "Could not detect PyTorch's pybind11 version")
    endif()
endfunction()

function(check_pybind11_differences)
    message(STATUS "pybind11 include dir: ${pybind11_SOURCE_DIR}/include/pybind11")
    message(STATUS "TORCH_PYBIND11_INCLUDE_DIR: ${TORCH_PYBIND11_INCLUDE_DIR}")

    execute_process(
        COMMAND diff -r "${TORCH_PYBIND11_INCLUDE_DIR}/pybind11" "${pybind11_SOURCE_DIR}/include/pybind11"
        RESULT_VARIABLE diff_result
        OUTPUT_VARIABLE diff_output
        ERROR_VARIABLE diff_error
    )

    if(diff_result EQUAL 0)
        message(STATUS "No differences found between the pybind11 directories")
    else()
        message(WARNING "Differences found between the pybind11 directories:")
        message(STATUS "${diff_output}")
        if(diff_error)
            message(STATUS "Diff errors: ${diff_error}")
        endif()
    endif()
endfunction()

find_package(pybind11)
if (pybind11_FOUND)
    message(STATUS "pybind11 found: ${pybind11_INCLUDE_DIRS}")
else()
    # Detect the version
    detect_torch_pybind11_version()

    # Set variables needed by pybind11
    set(PYBIND11_NEWPYTHON ON)
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES})

    # Use CPM to fetch pybind11
    CPMAddPackage(
        NAME pybind11
        GITHUB_REPOSITORY pybind/pybind11
        VERSION ${PYBIND11_VERSION}
        GIT_TAG v${PYBIND11_VERSION}
        OPTIONS
        "PYBIND11_INSTALL ON"
        "PYBIND11_TEST OFF"
    )

    check_pybind11_differences()
endif()
