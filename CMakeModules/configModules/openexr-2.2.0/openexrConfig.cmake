#############################################################################
## Configure modules for openexr. 
## Use the COMPONENTS version of FIND_PACKAGE to get any of the following:
## Half 
#############################################################################

if (CMAKE_VERSION VERSION_LESS 3.0.0)
    message(FATAL_ERROR "openexr requires at least CMake version 2.8.3")
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
	message(FATAL_ERROR "openexr not ready for this compiler")
endif()


get_filename_component(openexr_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/${RUNTIME}-${ARCH}/" ABSOLUTE)

macro(_check_file_exists file)
    if(NOT EXISTS "${file}" )
        message(FATAL_ERROR "The imported target references the file
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


macro(_populate_target_properties target Configuration IMPLIB_LOCATION LIB_LOCATION )
    set_property(TARGET ${target} APPEND PROPERTY IMPORTED_CONFIGURATIONS ${Configuration})

    set(imported_location "${openexr_INSTALL_PREFIX}/${LIB_LOCATION}")
    _check_file_exists(${imported_location})
	
    set_target_properties( ${target}  PROPERTIES
        "INTERFACE_LINK_LIBRARIES" "${_LIB_DEPENDENCIES}"
		"IMPORTED_LINK_INTERFACE_LIBRARIES_${Configuration}" ""
        "IMPORTED_LOCATION_${Configuration}" ${imported_location}
    )

    if(NOT "${LIB_LOCATION}" STREQUAL "") #add bin location
	    set(imported_implib "${openexr_INSTALL_PREFIX}/${IMPLIB_LOCATION}")
		_check_file_exists(${imported_implib})
        set_target_properties( ${target}  PROPERTIES
			"IMPORTED_IMPLIB_${Configuration}" ${imported_implib}
        )
    endif()
	
endmacro()

SET(openexr_COMPONENTS Half)

IF ( NOT openexr_FIND_COMPONENTS)
	SET(openexr_FIND_COMPONENTS ${openexr_COMPONENTS})
endif()


#####################################################################get requested targets
SET(openexr_LIBS "")
foreach(_component ${openexr_FIND_COMPONENTS})

	LIST(FIND openexr_COMPONENTS ${_component} _idx)

	if (_idx GREATER -1)
	
		if (NOT TARGET openexr::${_component})

			add_library(openexr::${_component} SHARED IMPORTED)
		
			#set relase properties
			 _populate_target_properties(openexr::${_component} RELEASE "lib/${_component}.lib" "lib/${_component}.dll" )

			#set debug properties
			#_populate_target_properties(openexr::${_component} DEBUG "lib/${_component}d.lib" "lib/${_component}d.dll" )
			
		endif()
		  
		set(openexr_${_component}_FOUND TRUE)
		LIST(APPEND openexr_LIBS openexr::${_component})
		
	else()
		MESSAGE(FATAL_ERROR "${_component} is not a component of openexr")
	endif()
endforeach()


_check_file_exists(${openexr_INSTALL_PREFIX}/include)
include_directories(${openexr_INSTALL_PREFIX}/include)