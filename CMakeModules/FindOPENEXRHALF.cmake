#
# Try to find OPENEXRHALF library and include path.
# Once done this will define
#
# OPENEXRHALF_FOUND
# OPENEXRHALF_INCLUDE_PATH
# OPENEXRHALF_LIBRARY
# 

IF (WIN32)
  # include
	FIND_PATH( OPENEXRHALF_INCLUDE_PATH OpenEXR/half.h
		$ENV{OPENEXRHALF_DIR}
		DOC "The directory where Half.h resides")

  # include
  IF( CMAKE_CL_64 )
    FIND_LIBRARY( OPENEXRHALF_LIBRARY
      NAMES Half
      PATHS
      $ENV{OPENEXRHALF_DIR}/lib/x64/Release
      DOC "The Half library")

    FIND_LIBRARY( OPENEXRHALF_LIBRARY_DEBUG
      NAMES Half
      PATHS
      $ENV{OPENEXRHALF_DIR}/lib/x64/Debug
      DOC "The Half library")
    
  ELSE( CMAKE_CL_64 )
    FIND_LIBRARY( OPENEXRHALF_LIBRARY
      NAMES Half
      PATHS
      $ENV{OPENEXRHALF_DIR}/lib/win32/Release
      DOC "The Half library")

    FIND_LIBRARY( OPENEXRHALF_LIBRARY_DEBUG
      NAMES Half
      PATHS
      $ENV{OPENEXRHALF_DIR}/lib/win32/Debug
      DOC "The Half library")
  
  ENDIF( CMAKE_CL_64 )
ELSE (WIN32)
  # TODO
ENDIF (WIN32)

 if(NOT OPENEXRHALF_LIBRARY_DEBUG)
    # There is no debug library
    set(OPENEXRHALF_LIBRARY_DEBUG ${OPENEXRHALF_LIBRARY})
    set(OPENEXRHALF_LIBRARIES     ${OPENEXRHALF_LIBRARY})
 else()
    # There IS a debug library
    set(OPENEXRHALF_LIBRARIES
        optimized ${OPENEXRHALF_LIBRARY}
        debug     ${OPENEXRHALF_LIBRARY_DEBUG}
    )
 endif()

IF (OPENEXRHALF_INCLUDE_PATH)
	SET( OPENEXRHALF_FOUND 1 CACHE STRING "Set to 1 if OPENEXRHALF is found, 0 otherwise")
ELSE (OPENEXRHALF_INCLUDE_PATH)
	SET( OPENEXRHALF_FOUND 0 CACHE STRING "Set to 1 if OPENEXRHALF is found, 0 otherwise")
ENDIF (OPENEXRHALF_INCLUDE_PATH)

MARK_AS_ADVANCED( OPENEXRHALF_FOUND )
