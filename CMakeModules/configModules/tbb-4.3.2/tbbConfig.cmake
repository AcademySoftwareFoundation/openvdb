get_filename_component(fullFolderName "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(folderName "${fullFolderName}" NAME)
string(REGEX REPLACE "([^-]*)-.*$" "\\1" PACKAGE_NAME ${folderName})
string(TOUPPER ${PACKAGE_NAME} UPPERCASE_PACKAGE_NAME)
string(REGEX REPLACE "[^-]*-(.*)$" "\\1" PACKAGE_VERSION ${folderName})

if (CMAKE_VERSION VERSION_LESS 3.0.0)
    message(FATAL_ERROR "${PACKAGE_NAME} requires at least CMake version 3.0.0")
endif()


IF (CMAKE_CL_64)
	SET(ARCH "intel64/")
ELSE()
	SET(ARCH "ia32/")
endif()

if(MSVC)
	IF(MSVC90)
		SET(RUNTIME "vc9")
	ELSEIF(MSVC10)
		SET(RUNTIME "vc10")
	ELSEIF(MSVC11)
		SET(RUNTIME "vc11")
	ELSEIF(MSVC12)
		SET(RUNTIME "vc12")
	ENDIF(MSVC90)
else()
	message(FATAL_ERROR "${PACKAGE_NAME} not ready for this compiler")
endif()

#if(NOT ${PACKAGE_NAME}_FIND_QUIETLY)
#  message(STATUS "${PACKAGE_NAME} ARCH: ${ARCH}")
#  message(STATUS "${PACKAGE_NAME} RUNTIME: ${RUNTIME}")
#endif()

get_filename_component(_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)


macro(_check_file_exists file)
    if(NOT EXISTS "${file}" )
        message(FATAL_ERROR "The imported target \"${PACKAGE_NAME}\" references the file
   \"${file}\"
but this file does not exist.  Possible reasons include:
* The file was deleted, renamed, or moved to another location.
* An install or uninstall procedure did not complete successfully.
* The installation package was faulty and contained
   \"${CMAKE_CURRENT_LIST_FILE}\"
but not all the files it references.
")
    endif()
endmacro()


macro(_populate_target_properties Configuration IMPLIB_LOCATION LIB_LOCATION )
    set_property(TARGET ${PACKAGE_NAME} APPEND PROPERTY IMPORTED_CONFIGURATIONS ${Configuration})

    set(imported_location "${_INSTALL_PREFIX}/${LIB_LOCATION}")
    _check_file_exists(${imported_location})
	
    set_target_properties(${PACKAGE_NAME} PROPERTIES
        "INTERFACE_LINK_LIBRARIES" "${_${PACKAGE_NAME}_LIB_DEPENDENCIES}"
		"IMPORTED_LINK_INTERFACE_LIBRARIES_${Configuration}" ""
        "IMPORTED_LOCATION_${Configuration}" ${imported_location}
    )

    if(NOT "${LIB_LOCATION}" STREQUAL "") #add bin location
	    set(imported_implib "${_INSTALL_PREFIX}/${IMPLIB_LOCATION}")
		_check_file_exists(${imported_implib})
        set_target_properties(${PACKAGE_NAME} PROPERTIES
			"IMPORTED_IMPLIB_${Configuration}" ${imported_implib}
        )
    endif()
	
endmacro()

if (NOT TARGET ${PACKAGE_NAME})

	add_library(${PACKAGE_NAME} SHARED IMPORTED)

	SET(_${PACKAGE_NAME}_LIB_DEPENDENCIES "")
	
	#set relase properties
	 _populate_target_properties(RELEASE "/lib/${ARCH}${RUNTIME}/${PACKAGE_NAME}.lib" "bin/${ARCH}${RUNTIME}/${PACKAGE_NAME}.dll" )

	#set debug properties
	_populate_target_properties(DEBUG "lib/${ARCH}${RUNTIME}/${PACKAGE_NAME}_debug.lib" "bin/${ARCH}${RUNTIME}/${PACKAGE_NAME}_debug.dll" )
	
endif()


set(${PACKAGE_NAME}_INCLUDE_DIRS "${_INSTALL_PREFIX}/include")
set(${PACKAGE_NAME}_LIBRARIES ${PACKAGE_NAME})



_check_file_exists(${${PACKAGE_NAME}_INCLUDE_DIRS})
include_directories(${${PACKAGE_NAME}_INCLUDE_DIRS})
	   
set(${PACKAGE_NAME}_FOUND TRUE)
	