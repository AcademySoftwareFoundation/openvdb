# Copyright (c) 2012-2016 DreamWorks Animation LLC
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DreamWorks Animation nor the names of
# its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
# LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
#

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
