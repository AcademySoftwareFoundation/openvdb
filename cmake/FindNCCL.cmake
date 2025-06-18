include (FindPackageHandleStandardArgs)

if(NCCL_ROOT)
  set(_NCCL_ROOT ${NCCL_ROOT})
elseif(DEFINED ENV{NCCL_ROOT})
  set(_NCCL_ROOT $ENV{NCCL_ROOT})
endif()

set(_NCCL_INCLUDE_SEARCH_DIRS "")
list(APPEND _NCCL_INCLUDE_SEARCH_DIRS
  ${_NCCL_ROOT}/include
)
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS ${_NCCL_INCLUDE_SEARCH_DIRS}
)

set(_NCCL_LIBRARY_SEARCH_DIRS "")
list(APPEND _NCCL_LIBRARY_SEARCH_DIRS
  ${_NCCL_ROOT}/bin
)
find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS ${_NCCL_LIBRARY_SEARCH_DIRS}
)

find_package_handle_standard_args(NCCL REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY)

if (NCCL_FOUND)
    mark_as_advanced(NCCL_INCLUDE_DIR)
    mark_as_advanced(NCCL_LIBRARY)
endif()

if (NCCL_FOUND AND NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL SHARED IMPORTED)
    if(MSVC)
      string(REPLACE ".lib" ".dll" NCCL_DLL ${NCCL_LIBRARY})
      set_property(TARGET NCCL::NCCL PROPERTY IMPORTED_LOCATION ${NCCL_DLL})
      set_property(TARGET NCCL::NCCL PROPERTY IMPORTED_IMPLIB ${NCCL_LIBRARY})
    else()
      set_property(TARGET NCCL::NCCL PROPERTY IMPORTED_LOCATION ${NCCL_LIBRARY})
    endif()
    target_include_directories(NCCL::NCCL INTERFACE ${NCCL_INCLUDE_DIR})
endif()
