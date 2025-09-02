include(CheckCXXCompilerFlag)

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/zlib/doc")
  message(FATAL_ERROR "Zlib not found. The project has to be cloned using --recursive (or call git submodule update --init --recursive) ")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)

  if (HAS_CPP14_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif (HAS_CPP11_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(FATAL_ERROR "C++11 support is needed (for pybind)!")
  endif()
endif()

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

# ZLib
set(ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
set(ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
add_subdirectory(zlib)

set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/zlib")
set(ZLIB_INCLUDE_DIR_SOURCE "${ZLIB_INCLUDE_DIR}")
set(ZLIB_INCLUDE_DIR_BINARY "${CMAKE_CURRENT_BINARY_DIR}/zlib")
set(ZLIB_LIBRARY zlibstatic)
set(ZLIB_FOUND TRUE)
set_property(TARGET ${ZLIB_LIBRARY} PROPERTY FOLDER "dependencies")
target_include_directories(${ZLIB_LIBRARY} INTERFACE
  $<BUILD_INTERFACE:${ZLIB_INCLUDE_DIR_SOURCE}>
  $<BUILD_INTERFACE:${ZLIB_INCLUDE_DIR_BINARY}>
)

# Use install rpath from this point onwards
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Build Thread Building Blocks (main shared libraries only)
set(TBB_TEST OFF CACHE BOOL "Enable testing" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "Enable examples" FORCE)
set(TBB_STRICT OFF CACHE BOOL "Treat compiler warnings as errors" FORCE)
set(TBB_WINDOWS_DRIVER OFF CACHE BOOL "Build as Universal Windows Driver (UWD)" FORCE)
set(TBB_NO_APPCONTAINER OFF CACHE BOOL "Apply /APPCONTAINER:NO (for testing binaries for Windows Store)" FORCE)
set(TBB4PY_BUILD OFF CACHE BOOL "Enable tbb4py build" FORCE)
set(TBB_BUILD ON CACHE BOOL "Enable tbb build" FORCE)
set(TBBMALLOC_BUILD OFF CACHE BOOL "Enable tbbmalloc build" FORCE)
set(TBBMALLOC_PROXY_BUILD OFF CACHE BOOL "Enable tbbmalloc_proxy build" FORCE)
set(TBB_CPF OFF CACHE BOOL "Enable preview features of the library" FORCE)
set(TBB_FIND_PACKAGE OFF CACHE BOOL "Enable search for external oneTBB using find_package instead of build from sources" FORCE)
set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ${CAKE_CROSSCOMPILING} CACHE BOOL "Disable HWLOC automatic search by pkg-config tool" FORCE)
set(TBB_ENABLE_IPO ON CACHE BOOL "Enable Interprocedural Optimization (IPO) during the compilation" FORCE)
set(TBB_FUZZ_TESTING OFF CACHE BOOL "Enable fuzz testing" FORCE)
set(TBB_INSTALL OFF CACHE BOOL "Enable installation" FORCE)
add_subdirectory(tbb)
set_property(TARGET tbb PROPERTY FOLDER "dependencies")

# Workaround for linking of TBB to Mitsuba 0.6 (our custom plugins)
if (MSVC)
  get_target_property(tbb_OUTPUT_DIR tbb LIBRARY_OUTPUT_DIRECTORY)

  get_target_property(tbb_NAME tbb OUTPUT_NAME)
  string(REGEX REPLACE "[0-9]+" "" tbb_NAME "${tbb_NAME}")

  string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)
  get_target_property(tbb_POSTFIX tbb "${U_CMAKE_BUILD_TYPE}_POSTFIX")
  if (tbb_POSTFIX)
    set(tbb_NAME "${tbb_NAME}${tbb_POSTFIX}")
  endif()

  set(TBB_COPY_OUTPUT_FILE "${tbb_OUTPUT_DIR}/${tbb_NAME}.lib")

  add_custom_command(OUTPUT ${TBB_COPY_OUTPUT_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_LINKER_FILE:tbb>"
    "${TBB_COPY_OUTPUT_FILE}"
    DEPENDS tbb
  )
  # Create target which consume the command via DEPENDS.
  add_custom_target(tbb-copy ALL
    DEPENDS ${TBB_COPY_OUTPUT_FILE}
  )
endif()

# Build GLFW
if(OPENVDB_BUILD_VDB_VIEW)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
  set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
  set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
  set(BUILD_SHARED_LIBS ${NANOGUI_BUILD_SHARED} CACHE BOOL " " FORCE)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Quench annoying deprecation warnings when compiling GLFW on OSX
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
  endif()

  add_subdirectory(glfw)
  set_property(TARGET glfw PROPERTY FOLDER "dependencies")

  set(GLFW3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include)
  set(GLFW3_INCLUDE_DIR ${GLFW3_INCLUDE_DIR} PARENT_SCOPE)

  set(GLFW3_glfw_LIBRARY glfw)
  set(GLFW3_glfw_LIBRARY ${GLFW3_glfw_LIBRARY} PARENT_SCOPE)
endif()


# Build Blosc
set(DEACTIVATE_AVX2          ON  CACHE BOOL " " FORCE)
set(PREFER_EXTERNAL_ZLIB  OFF CACHE BOOL " " FORCE)
set(BUILD_TESTS               OFF CACHE BOOL " " FORCE)
set(BUILD_BENCHMARKS          OFF CACHE BOOL " " FORCE)

add_subdirectory(c-blosc)
set_property(TARGET blosc_static blosc_shared PROPERTY FOLDER "dependencies")
set(BLOSC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/c-blosc/blosc)
set(BLOSC_INCLUDE_DIR ${BLOSC_INCLUDE_DIR} PARENT_SCOPE)
set(BLOSC_LIBRARY blosc_static PARENT_SCOPE)
set(BLOSC_blosc_LIBRARY blosc_static)
set(BLOSC_blosc_LIBRARY ${BLOSC_blosc_LIBRARY} PARENT_SCOPE)


# add_subdirectory(openvdb)
# set_property(TARGET openvdb_static openvdb_shared PROPERTY FOLDER "dependencies")
# set(OPENVDB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/include)
# set(OPENVDB_INCLUDE_DIRS ${OPENVDB_INCLUDE_DIRS} PARENT_SCOPE)

# Aliases for libraries used later on
add_library(Blosc::blosc ALIAS blosc_static)
add_library(ZLIB::ZLIB ALIAS ${ZLIB_LIBRARY})
