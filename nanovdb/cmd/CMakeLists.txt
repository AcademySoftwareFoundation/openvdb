option(NANOVDB_USE_EGL "Build with EGL support" OFF)
option(NANOVDB_USE_NATIVEFILEDIALOG "Build with nativefiledialog support" OFF)
option(NANOVDB_BUILD_INTERACTIVE_RENDERER "Build interactive renderer" ON)

# -----------------------------------------------------------------------
# workaround for win32 bug when nvcc "--keep" is used.
if(WIN32 AND NANOVDB_CUDA_KEEP_PTX)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
endif()

# -----------------------------------------------------------------------
# GLFW
#

set(IMGUI_SOURCE_FILES "")

if(NANOVDB_BUILD_INTERACTIVE_RENDERER)

  if(NOT USE_EMSCRIPTEN_TOOLCHAIN)
    # -----------------------------

    # wraps find_package(glfw3) and sets the glfw target
    # include(OpenVDBGLFW3Setup)

    if(glfw3_FOUND)
      set(NANOVDB_GLFW_LINK_LIBRARY "glfw")
      set(NANOVDB_USE_GLFW_FLAG "NANOVDB_USE_GLFW")
    elseif(NANOVDB_ALLOW_FETCHCONTENT)
      if(NOT glfw_POPULATED)
        message(STATUS "Downloading glfw...")
        FetchContent_Declare(
          glfw
          GIT_REPOSITORY https://github.com/glfw/glfw
          GIT_TAG 3.3)

        FetchContent_GetProperties(glfw)
        if(NOT glfw_POPULATED)
          FetchContent_Populate(glfw)
          set(GLFW_BUILD_EXAMPLES
              OFF
              CACHE INTERNAL "Build the GLFW example programs")
          set(GLFW_BUILD_TESTS
              OFF
              CACHE INTERNAL "Build the GLFW test programs")
          set(GLFW_BUILD_DOCS
              OFF
              CACHE INTERNAL "Build the GLFW documentation")
          set(GLFW_INSTALL
              OFF
              CACHE INTERNAL "Generate installation target")
          add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})
        endif()
        set(NANOVDB_GLFW_LINK_LIBRARY "glfw")
        set(NANOVDB_USE_GLFW_FLAG "NANOVDB_USE_GLFW")
      endif()
    endif()

    # -----------------------------
    if(NANOVDB_USE_NATIVEFILEDIALOG)
      if(NANOVDB_ALLOW_FETCHCONTENT)
        if(NOT nfd_POPULATED)
          message(STATUS "Downloading nfd...")

          FetchContent_Declare(
            nfd
            GIT_REPOSITORY https://github.com/mlabbe/nativefiledialog.git
            GIT_TAG master)

          FetchContent_GetProperties(nfd)
          if(NOT nfd_POPULATED)
            FetchContent_Populate(nfd)

            if(APPLE)
              list(APPEND NFD_SOURCE_FILES ${nfd_SOURCE_DIR}/src/nfd_cocoa.m
                   ${nfd_SOURCE_DIR}/src/nfd_common.c)
            elseif(WIN32)
              list(APPEND NFD_SOURCE_FILES ${nfd_SOURCE_DIR}/src/nfd_win.cpp
                   ${nfd_SOURCE_DIR}/src/nfd_common.c)
            elseif(UNIX)
              list(APPEND NFD_SOURCE_FILES ${nfd_SOURCE_DIR}/src/nfd_gtk.c
                   ${nfd_SOURCE_DIR}/src/nfd_common.c)
            endif()

            if(NFD_SOURCE_FILES)
              set(NANOVDB_USE_NFD_FLAG "NANOVDB_USE_NFD")

              # message(STATUS "NFD_SOURCE_FILES = ${NFD_SOURCE_FILES}")

              add_library(nfd ${NFD_SOURCE_FILES})

              target_include_directories(
                nfd PRIVATE ${nfd_SOURCE_DIR}/src ${nfd_SOURCE_DIR}/src/include)

              if(UNIX AND NOT APPLE)
                # Use the package PkgConfig to detect GTK+ headers/library files
                find_package(PkgConfig REQUIRED)
                pkg_check_modules(GTK REQUIRED gtk+-3.0)
                target_include_directories(nfd PRIVATE ${GTK_INCLUDE_DIRS})
                target_link_libraries(nfd ${GTK_LIBRARIES})
                target_compile_definitions(nfd PRIVATE ${GTK_CXXFLAGS_OTHER})
              endif()

              set(NANOVDB_NFD_LINK_LIBRARY "nfd")
              set(NANOVDB_NFD_INCLUDE_DIRECTORY ${nfd_SOURCE_DIR}/src/include)
            endif()
          endif()
        endif()
      endif()
    endif(NANOVDB_USE_NATIVEFILEDIALOG)
  else()
    # emscripten has built-in support for GLFW
    set(NANOVDB_USE_GLFW_FLAG "NANOVDB_USE_GLFW")
  endif(NOT USE_EMSCRIPTEN_TOOLCHAIN)

  # -----------------------------
  if(NANOVDB_ALLOW_FETCHCONTENT)
    if(NOT imgui_POPULATED)
      message(STATUS "Downloading imgui...")
      FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.79)

      FetchContent_GetProperties(imgui)
      if(NOT imgui_POPULATED)
        FetchContent_Populate(imgui)

        set(NANOVDB_USE_IMGUI_FLAG "NANOVDB_USE_IMGUI")

        list(
          APPEND
          IMGUI_SOURCE_FILES
          # ${imgui_SOURCE_DIR}/imgui_demo.cpp
          ${imgui_SOURCE_DIR}/imgui.cpp
          ${imgui_SOURCE_DIR}/imgui_draw.cpp
          ${imgui_SOURCE_DIR}/imgui_widgets.cpp
          ${imgui_SOURCE_DIR}/examples/imgui_impl_glfw.cpp
          ${imgui_SOURCE_DIR}/examples/imgui_impl_opengl3.h
          ${imgui_SOURCE_DIR}/examples/imgui_impl_opengl3.cpp)

        add_library(imgui ${IMGUI_SOURCE_FILES})

        target_include_directories(
          imgui PRIVATE ${imgui_SOURCE_DIR} ${NANOVDB_GLAD_INCLUDE_DIRECTORY})

        if(USE_EMSCRIPTEN_TOOLCHAIN)
          target_compile_definitions(imgui PRIVATE "-DIMGUI_IMPL_OPENGL_ES2")
        else()
          target_compile_definitions(imgui
                                     PRIVATE "-DIMGUI_IMPL_OPENGL_LOADER_GLAD")
        endif()

        target_link_libraries(imgui ${NANOVDB_GLFW_LINK_LIBRARY}
                              ${NANOVDB_GLAD_LINK_LIBRARY})

        set(NANOVDB_IMGUI "imgui")
      endif()
    endif(NOT imgui_POPULATED)
  endif()
endif(NANOVDB_BUILD_INTERACTIVE_RENDERER)

# -----------------------------------------------------------------------
if(TRUE)

  file(GLOB VIEWER_AGNOSTIC_CODE viewer/code/*)
  assign_source_group(${VIEWER_AGNOSTIC_CODE})
  file(GLOB VIEWER_OPTIX_CODE viewer/optix/*)
  assign_source_group(${VIEWER_OPTIX_CODE})

  list(
    APPEND
    VIEWER_SOURCES
    viewer/nanovdb_viewer
    viewer/RenderLauncherC99impl.c
    viewer/nanovdb_viewer.cpp
    viewer/FrameBuffer.h
    viewer/FrameBuffer.cpp
    viewer/FrameBufferHost.cpp
    viewer/FrameBufferHost.h
    viewer/Renderer.h
    viewer/Renderer.cpp
    viewer/BatchRenderer.h
    viewer/BatchRenderer.cpp
    viewer/RenderConstants.h
    viewer/RenderUtils.h
    viewer/RenderLevelSetUtils.h
    viewer/RenderFogVolumeUtils.h
    viewer/RenderGridUtils.h
    viewer/RenderPointsUtils.h
    viewer/RenderLauncherImpl.h
    viewer/RenderLauncher.h
    viewer/RenderLauncher.cpp
    viewer/RenderLauncherCpuMT.cpp
    viewer/AgnosticNanoVDB.h
    viewer/RenderLauncherC99.cpp)

  if(NOT (NANOVDB_GLFW_LINK_LIBRARY AND NANOVDB_GLAD_LINK_LIBRARY)
     AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
    message(WARNING " - GLFW not found for Viewer. Removing GLFW support.")
  else()
    list(APPEND VIEWER_SOURCES viewer/Viewer.cpp viewer/Viewer.h)
  endif()

  if(NOT (NANOVDB_GLAD_LINK_LIBRARY) AND NOT USE_EMSCRIPTEN_TOOLCHAIN)
    message(WARNING " - GLAD not found for Viewer. Removing GL support.")
  else()
    list(APPEND VIEWER_SOURCES viewer/FrameBufferGL.cpp viewer/FrameBufferGL.h
         viewer/RenderLauncherGL.cpp)
  endif()

  if(NOT CUDA_FOUND)
    message(WARNING " - CUDA not found for Viewer. Removing CUDA support.")
  else()
    list(APPEND VIEWER_SOURCES viewer/RenderLauncherCUDA.cu)
  endif()

  if(NOT OpenCL_FOUND)
    message(WARNING " - OpenCL not found for Viewer. Removing OpenCL support.")
  else()
    list(APPEND VIEWER_SOURCES viewer/RenderLauncherCL.cpp)
  endif()

  if(NOT OptiX_FOUND)
    message(WARNING " - OptiX not found for Viewer. Removing OptiX support.")
  else()
    list(APPEND VIEWER_SOURCES viewer/RenderLauncherOptix.cpp
         ${VIEWER_OPTIX_CODE})
  endif()

  add_executable(nanovdb_viewer ${VIEWER_SOURCES} ${VIEWER_AGNOSTIC_CODE})

  set_source_files_properties(${VIEWER_OPTIX_CODE} PROPERTIES HEADER_FILE_ONLY
                                                              TRUE)
  set_source_files_properties(${VIEWER_AGNOSTIC_CODE}
                              PROPERTIES HEADER_FILE_ONLY TRUE)

  target_include_directories(
    nanovdb_viewer
    PRIVATE ${NANOVDB_INCLUDE_DIR}
            ${TBB_INCLUDE_DIRECTORY} # optional
            ${CUDA_INCLUDE_DIRECTORY} # optional
            ${OPENCL_INCLUDE_DIRECTORY} # optional
            ${OPENGL_INCLUDE_DIRECTORY} # optional
            ${NANOVDB_OPTIX_INCLUDE_DIRECTORY} # optional
            ${imgui_SOURCE_DIR} # optional
            ${NANOVDB_GLAD_INCLUDE_DIRECTORY} # optional
            ${NANOVDB_OPENGLES_INCLUDE_DIR} # optional
            ${EGL_INCLUDE_DIRECTORY} # optional
            ${NANOVDB_NFD_INCLUDE_DIRECTORY} # optional
  )

  target_compile_definitions(
    nanovdb_viewer
    PRIVATE "-D${NANOVDB_USE_GLFW_FLAG}"
            "-D${NANOVDB_USE_GLAD_FLAG}"
            "-D${NANOVDB_USE_CUDA_FLAG}"
            "-D${NANOVDB_USE_OPENCL_FLAG}"
            "-D${NANOVDB_USE_OPTIX_FLAG}"
            "-DNANOVDB_OPTIX_RTC_OPTIX_DIR=\"${OptiX_INCLUDE_DIR}\""
            "-DNANOVDB_OPTIX_RTC_CUDA_DIR=\"${CUDA_INCLUDE_DIRECTORY}\""
            "-D${NANOVDB_USE_TBB_FLAG}"
            "-D${NANOVDB_USE_BLOSC_FLAG}"
            "-D${NANOVDB_USE_ZLIB_FLAG}"
            "-D${NANOVDB_USE_OPENVDB_FLAG}"
            "-D${NANOVDB_USE_IMGUI_FLAG}"
            "-D${NANOVDB_USE_OPENGL_FLAG}"
            "-D${NANOVDB_USE_EGL_FLAG}"
            "-D${NANOVDB_VIEWER_USE_GLES_FLAG}"
            "-D${NANOVDB_USE_NFD_FLAG}")

  target_link_libraries(
    nanovdb_viewer
    PRIVATE libnanovdb
            ${NANOVDB_GLAD_LINK_LIBRARY} # optional
            ${NANOVDB_GLFW_LINK_LIBRARY} # optional
            ${NANOVDB_IMGUI} # optional
            ${NANOVDB_OPENGL} # optional
            ${NANOVDB_OPENCL} # optional
            ${NANOVDB_TBB} # optional
            ${NANOVDB_BLOSC} # optional
            ${NANOVDB_ZLIB} # optional
            ${NANOVDB_OPENVDB} # optional
            ${NANOVDB_OPTIX}
            ${NVRTC_LINK_LIBRARY} # optional
            ${NANOVDB_OPENGLES_LIBRARIES} # optional
            ${NANOVDB_EGL} # optional
            ${NANOVDB_NFD_LINK_LIBRARY} # optional
  )

  if(APPLE)
    target_link_libraries(nanovdb_viewer PRIVATE "-framework OpenGL")
  endif()

  install(TARGETS nanovdb_viewer DESTINATION bin)

  if(USE_EMSCRIPTEN_TOOLCHAIN)

    set_target_properties(
      nanovdb_viewer
      PROPERTIES
        LINK_FLAGS
        "-O2 -s ASSERTIONS=1 -s SAFE_HEAP=1 -s USE_GLFW=3 -s USE_WEBGL2=1 -s ALLOW_MEMORY_GROWTH=1"
    )

    set(CMAKE_EXECUTABLE_SUFFIX ".html")
  endif()
endif()

# -----------------------------------------------------------------------
if(TRUE)

  if(NOT (OpenVDB_FOUND))
    message(WARNING " - OpenVDB required to build nanovdb_convert. Skipping.")
  else()
    add_executable(nanovdb_convert convert/nanovdb_convert.cpp)

    target_include_directories(nanovdb_convert PRIVATE ${NANOVDB_INCLUDE_DIR})

    target_compile_definitions(
      nanovdb_convert
      PRIVATE "-D${NANOVDB_USE_BLOSC_FLAG}" "-D${NANOVDB_USE_ZLIB_FLAG}"
              "-D${NANOVDB_USE_OPENVDB_FLAG}" "-D${NANOVDB_USE_TBB_FLAG}")

    target_link_libraries(
      nanovdb_convert PRIVATE libnanovdb ${NANOVDB_OPENVDB} ${NANOVDB_TBB}
                              ${NANOVDB_BLOSC} ${NANOVDB_ZLIB})

    install(TARGETS nanovdb_convert DESTINATION bin)

  endif()
endif()

# -----------------------------------------------------------------------
if(TRUE)

  add_executable(nanovdb_print print/nanovdb_print.cpp)

  target_include_directories(nanovdb_print PRIVATE ${NANOVDB_INCLUDE_DIR})

  target_compile_definitions(nanovdb_print PRIVATE "-D${NANOVDB_USE_BLOSC_FLAG}"
                                                   "-D${NANOVDB_USE_ZLIB_FLAG}")

  target_link_libraries(nanovdb_print PRIVATE libnanovdb ${NANOVDB_BLOSC}
                                              ${NANOVDB_ZLIB})

  install(TARGETS nanovdb_print DESTINATION bin)

endif()

# -----------------------------------------------------------------------
if(TRUE)

  add_executable(nanovdb_validate validate/nanovdb_validate.cpp)

  target_include_directories(nanovdb_validate PRIVATE ${NANOVDB_INCLUDE_DIR})

  target_compile_definitions(
    nanovdb_validate PRIVATE "-D${NANOVDB_USE_BLOSC_FLAG}"
                             "-D${NANOVDB_USE_ZLIB_FLAG}"
                             "-D${NANOVDB_USE_TBB_FLAG}")

  target_link_libraries(nanovdb_validate PRIVATE libnanovdb ${NANOVDB_BLOSC}
                                                 ${NANOVDB_ZLIB} ${NANOVDB_TBB})

  install(TARGETS nanovdb_validate DESTINATION bin)

endif()
