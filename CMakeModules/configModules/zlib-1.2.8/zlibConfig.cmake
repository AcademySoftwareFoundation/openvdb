
if (CMAKE_VERSION VERSION_LESS 2.8.3)
    message(FATAL_ERROR "zlib requires at least CMake version 2.8.3")
endif()

IF (CMAKE_CL_64)
	SET(ARCH "x64")
ELSE()
	SET(ARCH "x86")
endif()

if(MSVC)
	IF(MSVC90)
		SET(RUNTIME "msvc2008")
	ELSEIF(MSVC10)
		SET(RUNTIME "msvc2010")
	ELSEIF(MSVC11)
		SET(RUNTIME "msvc2012")
	ELSEIF(MSVC12)
		SET(RUNTIME "msvc2013")
	ENDIF(MSVC90)
else()
	message(FATAL_ERROR "zlib not ready for this compiler")
endif()

#if(NOT zlib_FIND_QUIETLY)
#  message(STATUS "zlib ARCH: ${ARCH}")
#  message(STATUS "zlib RUNTIME: ${RUNTIME}")
#endif()

get_filename_component(_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/${RUNTIME}-${ARCH}/" ABSOLUTE)


macro(_check_file_exists file)
    if(NOT EXISTS "${file}" )
        message(FATAL_ERROR "The imported target \"zlib\" references the file
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
    set_property(TARGET zlib APPEND PROPERTY IMPORTED_CONFIGURATIONS ${Configuration})

    set(imported_location "${_INSTALL_PREFIX}/${LIB_LOCATION}")
    _check_file_exists(${imported_location})
	
    set_target_properties(zlib PROPERTIES
        "INTERFACE_LINK_LIBRARIES" "${_zlib_LIB_DEPENDENCIES}"
		"IMPORTED_LINK_INTERFACE_LIBRARIES_${Configuration}" ""
        "IMPORTED_LOCATION_${Configuration}" ${imported_location}
    )

    if(NOT "${LIB_LOCATION}" STREQUAL "") #add bin location
	    set(imported_implib "${_INSTALL_PREFIX}/${IMPLIB_LOCATION}")
		_check_file_exists(${imported_implib})
        set_target_properties(zlib PROPERTIES
			"IMPORTED_IMPLIB_${Configuration}" ${imported_implib}
        )
    endif()
	
endmacro()

if (NOT TARGET zlib)

	add_library(zlib STATIC IMPORTED)

	SET(_zlib_LIB_DEPENDENCIES "")
	
	#set relase properties
	 _populate_target_properties(RELEASE "lib/zlibstatic.lib" "lib/zlibstatic.lib")

	#set debug properties
	#_populate_target_properties(DEBUG "lib/zlibstatic.lib" "")
	
endif()


set(zlib_INCLUDE_DIRS "${_INSTALL_PREFIX}/include")
set(zlib_LIBS zlib)


_check_file_exists(${zlib_INCLUDE_DIRS})
include_directories(${zlib_INCLUDE_DIRS})
	

   
set(zlib_FOUND TRUE)
	