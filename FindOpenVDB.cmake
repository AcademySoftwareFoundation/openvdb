# - Find OpenVDB
# Find the OpenVDB includes and client library
# This module defines
#  OpenVDB_INCLUDE_DIR, where to find openvdb/openvdb.h
#  OpenVDB_LIBRARIES, the libraries needed to use OpenVDB.
#  OpenVDB_BIN_DIRS, where the runtime files are present.
#  OpenVDB_FOUND, If false, do not try to use OpenVDB.


############ required components with header and library if COMPONENTS unspecified	
SET(OpenVDB_ALL_COMPONENTS Core Viewer)
IF (NOT OpenVDB_FIND_COMPONENTS)
	#default
	SET(OpenVDB_FIND_COMPONENTS ${OpenVDB_ALL_COMPONENTS})
ENDIF (NOT OpenVDB_FIND_COMPONENTS)


if(OpenVDB_INCLUDE_DIRS AND OpenVDB_LIBRARIES)
   set(OpenVDB_FOUND TRUE)

else(OpenVDB_INCLUDE_DIRS AND OpenVDB_LIBRARIES)

  #typical root dirs of installations, exactly one of them is used
  SET (OpenVDB_POSSIBLE_ROOT_DIRS  	${OpenVDB_DIR}
									$ENV{OpenVDB_DIR}
									"$ENV{ProgramFiles}/OpenVDB" )

  #select exactly ONE base directory/tree to avoid mixing different version headers and libs
  FIND_PATH(OpenVDB_ROOT_DIR NAMES include/openvdb/openvdb.h   PATHS ${OpenVDB_POSSIBLE_ROOT_DIRS})


  find_path(OpenVDB_INCLUDE_DIRS openvdb/openvdb.h  ${OpenVDB_ROOT_DIR}/include  )
  
  if(WIN32)
	find_path(OpenVDB_BIN_DIRS OPENVDBLIB.dll  ${OpenVDB_ROOT_DIR}/bin/Release ${OpenVDB_ROOT_DIR}/lib/Release  )
  endif(WIN32 )

  if(OpenVDB_INCLUDE_DIRS)
	

	FOREACH(_libName ${OpenVDB_FIND_COMPONENTS})
			
		#find the debug library
		FIND_LIBRARY(OpenVDB_${_libName}_DEBUG_LIBRARY NAMES "openvdb${_libName}d" PATHS ${OpenVDB_ROOT_DIR} PATH_SUFFIXES lib/Debug  NO_CMAKE_SYSTEM_PATH )
		#find the release library
		FIND_LIBRARY(OpenVDB_${_libName}_RELEASE_LIBRARY NAMES "openvdb${_libName}" PATHS ${OpenVDB_ROOT_DIR} PATH_SUFFIXES lib/Release  NO_CMAKE_SYSTEM_PATH )
			
		#Remove the cache value
		SET(OpenVDB_${_libName}_LIBRARY "" CACHE STRING "" FORCE)
		
			
		#both debug/release
		if(OpenVDB_${_libName}_DEBUG_LIBRARY AND OpenVDB_${_libName}_RELEASE_LIBRARY)
		
			SET(OpenVDB_${_libName}_LIBRARY  debug 		${OpenVDB_${_libName}_DEBUG_LIBRARY} 	
											 optimized 	${OpenVDB_${_libName}_RELEASE_LIBRARY}
				CACHE STRING "" FORCE)

				#only debug
		elseif(OpenVDB_${_libName}_DEBUG_LIBRARY)
			SET(OpenVDB_${_libName}_LIBRARY ${OpenVDB_${_libName}_DEBUG_LIBRARY}  CACHE STRING "" FORCE)
		#only release
		elseif( OpenVDB_${_libName}_RELEASE_LIBRARY)
			SET(OpenVDB_${_libName}_LIBRARY  ${OpenVDB_${_libName}_RELEASE_LIBRARY}  CACHE STRING "" FORCE)
		#no library found
		else()
			message(STATUS "WARNING: OPENVDB${_libName} was not found.")
			SET(OpenVDB_FOUND false)
		endif()
		
		LIST(APPEND OpenVDB_LIBRARIES ${OpenVDB_${_libName}_LIBRARY})
	
	ENDFOREACH(_libName ${OpenVDB_FIND_COMPONENTS})

	message(STATUS "Found OpenVDB: ${OpenVDB_INCLUDE_DIRS}, ${OpenVDB_LIBRARIES}")
	
	

	
  else(OpenVDB_INCLUDE_DIRS)
    set(OpenVDB_FOUND FALSE)
    if (OpenVDB_FIND_REQUIRED)
		message(FATAL_ERROR "OpenVDB not found.")
	else (OpenVDB_FIND_REQUIRED)
		message(STATUS "OpenVDB not found.")
	endif (OpenVDB_FIND_REQUIRED)
  endif(OpenVDB_INCLUDE_DIRS)

  mark_as_advanced(OpenVDB_INCLUDE_DIRS OpenVDB_LIBRARIES)

endif(OpenVDB_INCLUDE_DIRS AND OpenVDB_LIBRARIES)