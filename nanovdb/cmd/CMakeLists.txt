
# -----------------------------------------------------------------------
# workaround for win32 bug when nvcc "--keep" is used.
if(WIN32 AND NANOVDB_CUDA_KEEP_PTX)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
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
    nanovdb_validate
    PRIVATE "-D${NANOVDB_USE_BLOSC_FLAG}" "-D${NANOVDB_USE_ZLIB_FLAG}"
            "-D${NANOVDB_USE_TBB_FLAG}")

  target_link_libraries(nanovdb_validate PRIVATE libnanovdb ${NANOVDB_BLOSC}
                                                 ${NANOVDB_ZLIB} ${NANOVDB_TBB})

  install(TARGETS nanovdb_validate DESTINATION bin)

endif()
