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
# - Find Maya
#
# Author : Nicholas Yue yue.nicholas@gmail.com
#
# This auxiliary CMake file helps in find the MAYA headers and libraries
#
# MAYA_FOUND                  set if MAYA is found.
# MAYA_INCLUDE_DIR            MAYA's include directory
# MAYA_maya_LIBRARY            MAYA libraries
# MAYA_maya_preview_LIBRARY    MAYA_preview libraries (Mulitple Rendering Context)
# MAYA_mayamalloc_LIBRARY      MAYAmalloc libraries (Mulitple Rendering Context)

FIND_PACKAGE ( PackageHandleStandardArgs )

FIND_PATH( MAYA_LOCATION include/maya/MTypes.h
  "$ENV{MAYA_LOCATION}"
  NO_DEFAULT_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  NO_CMAKE_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  NO_CMAKE_SYSTEM_PATH
  )

FIND_PACKAGE_HANDLE_STANDARD_ARGS ( Maya
  REQUIRED_VARS MAYA_LOCATION
  )

IF ( MAYA_FOUND )
  SET( MAYA_INCLUDE_DIR       "${MAYA_LOCATION}/include" CACHE STRING "Maya include directory")

  SET ( MAYA_BOOST_VERSION "1_52" )

  LIST( APPEND MAYA_LIBRARY_COMPONENTS  OpenMaya )
  LIST( APPEND MAYA_LIBRARY_COMPONENTS  OpenMayaAnim )
  LIST( APPEND MAYA_LIBRARY_COMPONENTS  OpenMayaFX )
  LIST( APPEND MAYA_LIBRARY_COMPONENTS  OpenMayaRender )
  LIST( APPEND MAYA_LIBRARY_COMPONENTS  OpenMayaUI )
  LIST( APPEND MAYA_LIBRARY_COMPONENTS  Foundation )
  LIST( APPEND MAYA_LIBRARY_COMPONENTS  tbb )
  LIST( APPEND MAYA_LIBRARY_COMPONENTS  glew )
  LIST( APPEND MAYA_LIBRARY_COMPONENTS  glewmx )
  
  FOREACH ( lib_component ${MAYA_LIBRARY_COMPONENTS} )
    FIND_LIBRARY ( MAYA_${lib_component}_LIBRARY  ${lib_component}
      PATHS ${MAYA_LOCATION}/lib
	  NO_DEFAULT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      )
  ENDFOREACH ()

  LIST ( APPEND MAYA_BOOST_LIBRARY_COMPONENTS awBoost_filesystem )
  LIST ( APPEND MAYA_BOOST_LIBRARY_COMPONENTS awBoost_python )
  LIST ( APPEND MAYA_BOOST_LIBRARY_COMPONENTS awBoost_regex )
  LIST ( APPEND MAYA_BOOST_LIBRARY_COMPONENTS awBoost_signals )
  LIST ( APPEND MAYA_BOOST_LIBRARY_COMPONENTS awBoost_system )
  LIST ( APPEND MAYA_BOOST_LIBRARY_COMPONENTS awBoost_thread )

  FOREACH ( lib_component ${MAYA_BOOST_LIBRARY_COMPONENTS} )
    FIND_LIBRARY ( MAYA_${lib_component}_LIBRARY  ${lib_component}-${MAYA_BOOST_VERSION}
      PATHS ${MAYA_LOCATION}/lib
	  NO_DEFAULT_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      )
  ENDFOREACH ()

  IF (APPLE)
    SET ( MAYA_DEFINITIONS
      -DMAC_PLUGIN
	  -DREQUIRE_IOSTREAM
	  -DOSMac_
	  -DOSMac_MachO_
	  -D_BOOL
      )
  ELSEIF (WIN32)
    SET ( MAYA_DEFINITIONS
      -DNOMINMAX
      -DNT_PLUGIN
      -DREQUIRE_IOSTREAM
      -D_USE_MATH_DEFINES
      -D_CRT_SECURE_NO_WARNINGS
      )
  ELSE (APPLE)
    SET ( MAYA_DEFINITIONS
      -DBits64_
      -m64
      -DUNIX
      -D_BOOL
      -DLINUX
      -DFUNCPROTO
      -DGL_GLEXT_PROTOTYPES=1
      -D_GNU_SOURCE
      -DLINUX_64
      -fPIC
      -fno-strict-aliasing
      -DREQUIRE_IOSTREAM
      -Wno-deprecated
      -Wall
      -Wno-multichar
      -Wno-comment
      -Wno-sign-compare
      -funsigned-char
      -Wno-reorder
      -pthread
      -fno-gnu-keywords
      )
  ENDIF ()

  MACRO( MAYA_SET_LIBRARY_PROPERTIES NAME )
    IF (WIN32)
      SET_TARGET_PROPERTIES ( ${NAME} PROPERTIES
		SUFFIX ".mll"
		PREFIX ""
		LINK_FLAGS "/export:initializePlugin /export:uninitializePlugin"
		)
    ELSEIF (APPLE)
      SET_TARGET_PROPERTIES ( ${NAME} PROPERTIES
		SUFFIX ".bundle"
		PREFIX "")
    ELSE ()
      SET_TARGET_PROPERTIES ( ${NAME} PROPERTIES
		PREFIX "")
    ENDIF ()
    
  ENDMACRO ()

ENDIF ( MAYA_FOUND )
