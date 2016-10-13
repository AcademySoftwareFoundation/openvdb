#-*-cmake-*-
# - Find GLFW3
#
# Author : Nicholas Yue yue.nicholas@gmail.com
#
# This auxiliary CMake file helps in find the glfw3 headers and libraries
#
# GLFW3_FOUND            set if glfw3 is found.
# GLFW3_INCLUDE_DIR      glfw3's include directory
# GLFW3_LIBRARY_DIR      glfw3's library directory
# GLFW3_LIBRARIES        all glfw3 libraries

FIND_PACKAGE ( PackageHandleStandardArgs )

FIND_PATH( GLFW3_LOCATION include/GLFW/glfw3.h
  "$ENV{GLFW3_ROOT}"
  NO_DEFAULT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS ( GLFW3
  REQUIRED_VARS GLFW3_LOCATION
  )

IF (GLFW3_FOUND)
  SET( GLFW3_INCLUDE_DIR "${GLFW3_LOCATION}/include" CACHE STRING "GLFW3 include path")
  IF (GLFW3_USE_STATIC_LIBS)
    FIND_LIBRARY ( GLFW3_glfw_LIBRARY  glfw3  ${GLFW3_LOCATION}/lib
	  NO_DEFAULT_PATH
	  NO_CMAKE_ENVIRONMENT_PATH
	  NO_CMAKE_PATH
	  NO_SYSTEM_ENVIRONMENT_PATH
	  NO_CMAKE_SYSTEM_PATH
	  )
  ELSE (GLFW3_USE_STATIC_LIBS)
    FIND_LIBRARY ( GLFW3_glfw_LIBRARY  glfw  ${GLFW3_LOCATION}/lib
	  NO_DEFAULT_PATH
	  NO_CMAKE_ENVIRONMENT_PATH
	  NO_CMAKE_PATH
	  NO_SYSTEM_ENVIRONMENT_PATH
	  NO_CMAKE_SYSTEM_PATH
	  )
  ENDIF (GLFW3_USE_STATIC_LIBS)

  IF (APPLE)
	FIND_LIBRARY ( COCOA_LIBRARY Cocoa )
	FIND_LIBRARY ( IOKIT_LIBRARY IOKit )
	FIND_LIBRARY ( COREVIDEO_LIBRARY CoreVideo )
  ELSEIF (UNIX AND NOT APPLE)
	SET ( GLFW3_REQUIRED_X11_LIBRARIES
      Xi
      Xrandr
      Xinerama
      Xcursor
      )
  ENDIF ()
  
  SET ( GLFW3_LIBRARIES
	${OPENGL_gl_LIBRARY}
	${OPENGL_glu_LIBRARY}
	${GLFW3_glfw_LIBRARY}
	# UNIX                                                                                                                      
	${GLFW3_REQUIRED_X11_LIBRARIES}
	# APPLE                                                                                                                     
	${COCOA_LIBRARY}
	${IOKIT_LIBRARY}
	${COREVIDEO_LIBRARY}
	CACHE STRING "GLFW3 required libraries"
	)
  
ENDIF ()
