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

# -*- cmake -*-
# - Find Blosc
#
# Author : Nicholas Yue yue.nicholas@gmail.com
#
# BLOSC_FOUND            set if Blosc is found.
# BLOSC_INCLUDE_DIR      Blosc's include directory
# BLOSC_LIBRARYDIR      Blosc's library directory
# BLOSC_LIBRARIES        all Blosc libraries

FIND_PACKAGE ( PackageHandleStandardArgs )

FIND_PATH( BLOSC_LOCATION include/blosc.h
  "$ENV{BLOSC_ROOT}"
  NO_DEFAULT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  PATHS ${SYSTEM_LIBRARY_PATHS}
  )

FIND_PACKAGE_HANDLE_STANDARD_ARGS ( Blosc
  REQUIRED_VARS BLOSC_LOCATION
  )

IF ( BLOSC_FOUND )

  SET ( BLOSC_LIBRARYDIR ${BLOSC_LOCATION}/lib
    CACHE STRING "Blosc library directories")

  SET ( _blosc_library_name "blosc" )

  # Static library setup
  IF (Blosc_USE_STATIC_LIBS)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES_BACKUP ${CMAKE_FIND_LIBRARY_SUFFIXES})
	IF (WIN32)
	  SET ( _blosc_library_name "libblosc" )
	ELSE ()
	  SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
	ENDIF ()
  ENDIF()

  FIND_LIBRARY ( BLOSC_blosc_LIBRARY ${_blosc_library_name}
    PATHS ${BLOSC_LIBRARYDIR}
    NO_DEFAULT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    )

  # Static library tear down
  IF (Blosc_USE_STATIC_LIBS)
    SET( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_BACKUP} )
  ENDIF()

  SET( BLOSC_INCLUDE_DIR "${BLOSC_LOCATION}/include" CACHE STRING "Blosc include directory" )

ENDIF ( BLOSC_FOUND )
