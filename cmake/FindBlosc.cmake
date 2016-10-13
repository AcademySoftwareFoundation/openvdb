# -*- cmake -*-
# - Find Blosc
#
# Author : Nicholas Yue yue.nicholas@gmail.com
#
# BLOSC_FOUND            set if Blosc is found.
# BLOSC_INCLUDE_DIR      Blosc's include directory
# BLOSC_LIBRARY_DIR      Blosc's library directory
# BLOSC_LIBRARIES        all Blosc libraries

FIND_PACKAGE ( PackageHandleStandardArgs )

FIND_PATH( BLOSC_LOCATION include/blosc.h
  "$ENV{BLOSC_ROOT}"
  NO_DEFAULT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  )

FIND_PACKAGE_HANDLE_STANDARD_ARGS ( Blosc
  REQUIRED_VARS BLOSC_LOCATION
  )

IF ( BLOSC_FOUND )
  
  # Static library setup
  IF (Blosc_USE_STATIC_LIBS)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES_BACKUP ${CMAKE_FIND_LIBRARY_SUFFIXES})
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  ENDIF()

  FIND_LIBRARY ( BLOSC_blosc_LIBRARY blosc
    PATHS ${BLOSC_LOCATION}/lib
    NO_DEFAULT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    )
  
  # Static library tear down
  IF (Blosc_USE_STATIC_LIBS)
    SET( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_BACKUP} )
  ENDIF()

  SET( BLOSC_INCLUDE_DIR "${BLOSC_LOCATION}/include" CACHE STRING "Blosc include directory" )

ENDIF ( BLOSC_FOUND )
