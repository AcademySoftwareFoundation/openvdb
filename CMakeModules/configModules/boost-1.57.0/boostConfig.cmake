
if (CMAKE_VERSION VERSION_LESS 2.8.3)
    message(FATAL_ERROR "boost requires at least CMake version 2.8.3")
endif()

get_filename_component(fullFolderName "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(folderName "${fullFolderName}" NAME)

string(REGEX REPLACE "([~-]*)-.*$" "\\1" NOCASE_PACKAGE_NAME ${folderName})
string(TOUPPER ${NOCASE_PACKAGE_NAME} PACKAGE_NAME)
string(REGEX REPLACE "[^-]*-(.*)$" "\\1" PACKAGE_VERSION ${folderName})

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
	ELSE()
		message(FATAL_ERROR "${PACKAGE_NAME} not ready for this Visual C++ version")
	ENDIF(MSVC90)
else()
	message(FATAL_ERROR "${PACKAGE_NAME} not ready for this compiler")
endif()

#if(NOT ${PACKAGE_NAME}_FIND_QUIETLY)
#  message(STATUS "${PACKAGE_NAME} ARCH: ${ARCH}")
#  message(STATUS "${PACKAGE_NAME} RUNTIME: ${RUNTIME}")
#endif()

#BOOST specific: Will call find_package again, setting Boost_NO_BOOST_CMAKE to 1 (so it won't look for boostConfig.cmake anymore)

IF (NOT DEFINED Boost_USE_STATIC_LIBS)
	set(Boost_USE_STATIC_LIBS ON)
ENDIF()

SET(Boost_NO_BOOST_CMAKE 1)
SET(BOOST_ROOT ${fullFolderName})
SET(BOOST_INCLUDEDIR ${fullFolderName}/include)
SET(BOOST_LIBRARYDIR ${fullFolderName}/lib-${RUNTIME}-${ARCH})
SET(Boost_FIND_QUIETLY true)

IF(DEFINED boost_FIND_COMPONENTS)
	find_package(Boost ${PACKAGE_VERSION} MODULE COMPONENTS  ${boost_FIND_COMPONENTS} )

	set(Boost_LIBRARIES "")
	FOREACH(_component_lowercase ${boost_FIND_COMPONENTS})
		string(TOUPPER ${_component_lowercase} _component)
		if (${Boost_${_component}_FOUND})
			set(_target Boost::${_component_lowercase})
			if (NOT TARGET ${_target})
				if (${Boost_USE_STATIC_LIBS}) 
					add_library(${_target} STATIC IMPORTED)
				else()
					add_library(${_target} SHARED IMPORTED)

				endif()
				FOREACH(_configuration DEBUG RELEASE RELWITHDEBINFO)
          IF(${_configuration} STREQUAL RELWITHDEBINFO)
            set(_static_lib_name ${Boost_${_component}_LIBRARY_RELEASE})
          ELSE()
            set(_static_lib_name ${Boost_${_component}_LIBRARY_${_configuration}})
          ENDIF()
        
        
					set_property(TARGET ${_target} APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_configuration})

					if (${Boost_USE_STATIC_LIBS}) 
						set_target_properties(${_target}  PROPERTIES
							"INTERFACE_LINK_LIBRARIES" ""
							"IMPORTED_LINK_INTERFACE_LIBRARIES_${_configuration}" ""
							"IMPORTED_LOCATION_${_configuration}" ${_static_lib_name}
						)
					else()
						string(REGEX REPLACE "(.*)\\.lib$" "\\1.dll" _dynamic_lib_name ${_static_lib_name})
					
						set_target_properties(${_target}  PROPERTIES
							"INTERFACE_LINK_LIBRARIES" ""
							"IMPORTED_LINK_INTERFACE_LIBRARIES_${_configuration}" ""
							"IMPORTED_LOCATION_${_configuration}" ${_dynamic_lib_name}
							"IMPORTED_IMPLIB_${_configuration}" ${_static_lib_name}
						)
					endif()
				ENDFOREACH()
				LIST(APPEND Boost_LIBRARIES ${_target})
			endif()
		else()
			MESSAGE(FATAL "Component ${_component_lowercase} not found.")
		endif()
	ENDFOREACH()
	#MESSAGE(STATUS "boost libraries are ${Boost_LIBRARIES}")	
else()
	find_package(${PACKAGE_NAME} ${PACKAGE_VERSION})
	MESSAGE(FATAL "Components not specified.")
endif()

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
