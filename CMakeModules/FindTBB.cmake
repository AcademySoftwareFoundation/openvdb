#
# Try to find TBB library and include path.
# Once done this will define
#
# TBB_FOUND
# TBB_INCLUDE_PATH
# TBB_LIBRARY
# 

IF (WIN32)
	FIND_PATH( TBB_INCLUDE_PATH tbb/tbb.h
		$ENV{TBB_DIR}/include
		DOC "The directory where tbb.h resides")
	FIND_LIBRARY( TBB_LIBRARY_RELEASE
		NAMES tbb
		PATHS
		$ENV{TBB_DIR}/build/vs2010/intel64/Release
		DOC "The tbb library")
	FIND_LIBRARY( TBB_LIBRARY_DEBUG
		NAMES tbb_debug
		PATHS
		$ENV{TBB_DIR}/build/vs2010/intel64/Debug
		DOC "The tbb library")
	
  #both debug/release
	if(TBB_LIBRARY_RELEASE AND TBB_LIBRARY_DEBUG)
		SET(TBB_LIBRARY debug ${TBB_LIBRARY_DEBUG} optimized ${TBB_LIBRARY_RELEASE})
	#only debug
	elseif(TBB_LIBRARY_DEBUG)
		SET(TBB_LIBRARY ${TBB_LIBRARY_DEBUG})
	#only release
	elseif(TBB_LIBRARY_RELEASE)
		SET(TBB_LIBRARY ${TBB_LIBRARY_RELEASE})
	#no library found
	else()
		message(STATUS "WARNING: TBB was not found.")
		SET(TBB_FOUND false)
	endif()
    
    
ELSE (WIN32)
  # TODO
ENDIF (WIN32)

IF (TBB_INCLUDE_PATH)
	SET( TBB_FOUND 1 CACHE STRING "Set to 1 if TBB is found, 0 otherwise")
ELSE (TBB_INCLUDE_PATH)
	SET( TBB_FOUND 0 CACHE STRING "Set to 1 if TBB is found, 0 otherwise")
ENDIF (TBB_INCLUDE_PATH)

MARK_AS_ADVANCED( TBB_FOUND )
