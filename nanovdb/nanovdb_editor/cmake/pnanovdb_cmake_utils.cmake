include_guard(GLOBAL)

function(pnanovdb_prebuild_link SOURCE_PATH TARGET_PATH)
  file(TO_NATIVE_PATH ${SOURCE_PATH} SOURCE_PATH)
  if(NOT EXISTS ${SOURCE_PATH})
    message(FATAL_ERROR "Source path ${SOURCE_PATH} does not exist")
  endif()
  file(TO_NATIVE_PATH ${TARGET_PATH} TARGET_PATH)
  if(NOT EXISTS ${TARGET_PATH})
    if(WIN32)
      execute_process(
        COMMAND cmd /c mklink /J "${TARGET_PATH}" "${SOURCE_PATH}"
      )
    else()
      message(STATUS "Creating link from ${SOURCE_PATH} to ${TARGET_PATH}")
      execute_process(
        COMMAND ln -sf "${SOURCE_PATH}" "${TARGET_PATH}" OUTPUT_QUIET
      )
    endif()
  endif()
endfunction()

