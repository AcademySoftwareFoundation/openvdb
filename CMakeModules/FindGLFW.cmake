#
# Try to find GLFW library and include path.
# Once done this will define
#
# GLFW_FOUND
# GLFW_INCLUDE_PATH
# GLFW_LIBRARY
# 

IF (WIN32)
	FIND_PATH(GLFW_INCLUDE_PATH GL/glfw.h
		$ENV{GLFW_DIR}/include
		DOC "The directory where GLFW.h resides")
	FIND_LIBRARY(GLFW_LIBRARY_RELEASE
		NAMES GLFW
		PATHS
		$ENV{GLFW_DIR}/support/msvc100/x64/Release
		DOC "The GLFW library (release)")
	FIND_LIBRARY(GLFW_LIBRARY_DEBUG
		NAMES GLFW
		PATHS
		$ENV{GLFW_DIR}/support/msvc100/x64/Debug
		DOC "The GLFW library (debug)")
	
  #both debug/release
	if(GLFW_LIBRARY_RELEASE AND GLFW_LIBRARY_DEBUG)
		SET(GLFW_LIBRARY debug ${GLFW_LIBRARY_DEBUG} optimized ${GLFW_LIBRARY_RELEASE})
	#only debug
	elseif(GLFW_LIBRARY_DEBUG)
		SET(GLFW_LIBRARY ${GLFW_LIBRARY_DEBUG})
	#only release
	elseif(GLFW_LIBRARY_RELEASE)
		SET(GLFW_LIBRARY ${GLFW_LIBRARY_RELEASE})
	#no library found
	else()
		message(STATUS "WARNING: GLFW was not found.")
		SET(GLFW_FOUND false)
	endif()
    
    
ELSE (WIN32)
  # TODO
ENDIF (WIN32)

IF (GLFW_INCLUDE_PATH)
	SET( GLFW_FOUND 1 CACHE STRING "Set to 1 if GLFW is found, 0 otherwise")
ELSE (GLFW_INCLUDE_PATH)
	SET( GLFW_FOUND 0 CACHE STRING "Set to 1 if GLFW is found, 0 otherwise")
ENDIF (GLFW_INCLUDE_PATH)

MARK_AS_ADVANCED( GLFW_FOUND )
