#
# Try to find ZLIB library and include path.
# Once done this will define
#
# ZLIB_FOUND
# ZLIB_INCLUDE_PATH
# ZLIB_LIBRARY
# 

IF (WIN32)
	FIND_PATH( ZLIB_INCLUDE_PATH zlib.h
		$ENV{ZLIB_DIR}
		DOC "The directory where zlib.h resides")
	FIND_LIBRARY( ZLIB_LIBRARY
		NAMES zlibwapi
		PATHS
		$ENV{ZLIB_DIR}/contrib/vstudio/vc10/x64/ZlibDllRelease
		DOC "The ZLIB library")
ELSE (WIN32)
  # TODO
ENDIF (WIN32)

IF (ZLIB_INCLUDE_PATH)
	SET( ZLIB_FOUND 1 CACHE STRING "Set to 1 if ZLIB is found, 0 otherwise")
ELSE (ZLIB_INCLUDE_PATH)
	SET( ZLIB_FOUND 0 CACHE STRING "Set to 1 if ZLIB is found, 0 otherwise")
ENDIF (ZLIB_INCLUDE_PATH)

MARK_AS_ADVANCED( ZLIB_FOUND )
