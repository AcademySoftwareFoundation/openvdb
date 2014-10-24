PROJECT(openvdbCore)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

SET(VERSION_H_IN  ${PROJECT_SOURCE_DIR}/../CMakeConf/version.h.in)
SET(VERSION_RC_IN ${PROJECT_SOURCE_DIR}/../CMakeConf/version.rc.in)

CONFIGURE_FILE( ${VERSION_RC_IN} ${PROJECT_BINARY_DIR}/version.rc)
CONFIGURE_FILE( ${VERSION_H_IN} ${PROJECT_BINARY_DIR}/version.h)

FILE(GLOB HDRS_FILES *.h ) 
FILE(GLOB HDRS_FILES_IO io/*.h)
FILE(GLOB HDRS_FILES_MATH math/*.h)
FILE(GLOB HDRS_FILES_METADATA metadata/*.h)  
FILE(GLOB HDRS_FILES_TOOLS tools/*.h)
FILE(GLOB HDRS_FILES_TREE tree/*.h)
FILE(GLOB HDRS_FILES_UTIL util/*.h)

FILE(GLOB SRCS_FILES *.cc)
FILE(GLOB SRCS_FILES_IO io/*.cc)
FILE(GLOB SRCS_FILES_MATH math/*.cc)
FILE(GLOB SRCS_FILES_METADATA metadata/*.cc)  
FILE(GLOB SRCS_FILES_TOOLS tools/*.cc)
FILE(GLOB SRCS_FILES_TREE tree/*.cc)
FILE(GLOB SRCS_FILES_UTIL util/*.cc)

SOURCE_GROUP("io" FILES ${HDRS_FILES_IO} ${SRCS_FILES_IO})
SOURCE_GROUP("math" FILES ${HDRS_FILES_MATH} ${SRCS_FILES_MATH})
SOURCE_GROUP("metadata" FILES ${HDRS_FILES_METADATA} ${SRCS_FILES_METADATA})
SOURCE_GROUP("tools" FILES ${HDRS_FILES_TOOLS} ${SRCS_FILES_TOOLS})
SOURCE_GROUP("tree" FILES ${HDRS_FILES_TREE} ${SRCS_FILES_TREE})
SOURCE_GROUP("util" FILES ${HDRS_FILES_UTIL} ${SRCS_FILES_UTIL})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_LIBRARY(${PROJECT_NAME}  SHARED
	${HDRS_FILES} ${HDRS_FILES_IO} ${HDRS_FILES_MATH} ${HDRS_FILES_METADATA} ${HDRS_FILES_TOOLS} ${HDRS_FILES_TREE} ${HDRS_FILES_UTIL}
	${SRCS_FILES} ${SRCS_FILES_IO} ${SRCS_FILES_MATH} ${SRCS_FILES_METADATA} ${SRCS_FILES_TOOLS} ${SRCS_FILES_TREE} ${SRCS_FILES_UTIL}  
  ${PROJECT_BINARY_DIR}/version.rc
)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_LIBRARIES} ${ZLIB_LIBRARY} ${OPENEXRHALF_LIBRARY} ${TBB_LIBRARY})

# /bigobj is required for windows builds because of the size of some object files (tree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/bigobj /DOPENVDB_PRIVATE")
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES VERSION ${OPENVDB_VERSION})
#SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES FOLDER OPENVDBLIB)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES STATIC_LIBRARY_FLAGS \"/machine:x64\")


############################## Install
INSTALL(TARGETS ${PROJECT_NAME} 
	RUNTIME DESTINATION bin COMPONENT Runtime
	LIBRARY DESTINATION lib COMPONENT Development
	ARCHIVE DESTINATION lib COMPONENT Development)
	
#copy header files to include directory
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include COMPONENT Development FILES_MATCHING PATTERN "*.h" )

######################################################################################################################################################
######################################################################################################################################################
PROJECT(openvdbViewer)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

# OpenGL
FIND_PACKAGE(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

FIND_PACKAGE(GLEW REQUIRED)
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_PATH})

FIND_PACKAGE(GLFW REQUIRED)
INCLUDE_DIRECTORIES(${GLFW_INCLUDE_PATH})


FILE(GLOB OPENVDBVIEWER_HDRS_FILES viewer/*.h ) 
FILE(GLOB OPENVDBVIEWER_SRCS_FILES viewer/*.cc)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/..)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_LIBRARY(${PROJECT_NAME} STATIC
	${OPENVDBVIEWER_HDRS_FILES}
	${OPENVDBVIEWER_SRCS_FILES}
  ${PROJECT_BINARY_DIR}/version.rc
)
ADD_DEPENDENCIES(${PROJECT_NAME} Core)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_LIBRARIES} ${ZLIB_LIBRARY} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENEXRHALF_LIBRARY} ${TBB_LIBRARY} openvdbCore)

# /bigobj is required for windows builds because of the size of
# some object files (tree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/bigobj")
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES VERSION ${OPENVDB_VERSION})
#SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES FOLDER OPENVDBVIEWER)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES STATIC_LIBRARY_FLAGS \"/machine:x64\")

############################## Install
INSTALL(TARGETS ${PROJECT_NAME} 
	RUNTIME DESTINATION bin COMPONENT Runtime
	LIBRARY DESTINATION lib COMPONENT Development
	ARCHIVE DESTINATION lib COMPONENT Development)
	
#copy header files to include directory
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include COMPONENT Development FILES_MATCHING PATTERN "*.h" )



######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
################################################  Project specification, add files directly here...  #################################################
PROJECT(vdbViewerApp)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

FILE(GLOB OPENVDBVIEWER_APP_HDRS_FILES cmd/openvdb_view/*.h ) 
FILE(GLOB OPENVDBVIEWER_SRCS_FILES cmd/openvdb_view/*.cc)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/..)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_EXECUTABLE(${PROJECT_NAME}
	${OPENVDBVIEWER_APP_HDRS_FILES}
	${OPENVDBVIEWER_SRCS_FILES}
  ${PROJECT_BINARY_DIR}/version.rc
)
ADD_DEPENDENCIES(${PROJECT_NAME} Core Viewer)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_LIBRARIES} ${ZLIB_LIBRARY} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENEXRHALF_LIBRARY} ${TBB_LIBRARY} openvdbCore openvdbViewer)

# /bigobj is required for windows builds because of the size of
# some object files (tree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/bigobj")
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES VERSION ${OPENVDB_VERSION})
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES FOLDER apps)

############################## Install
INSTALL(TARGETS ${PROJECT_NAME} 
	RUNTIME DESTINATION bin COMPONENT Runtime
	LIBRARY DESTINATION lib COMPONENT Development
	ARCHIVE DESTINATION lib COMPONENT Development)
	
#copy header files to include directory
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include COMPONENT Development FILES_MATCHING PATTERN "*.h" )


######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
################################################  Project specification, add files directly here...  #################################################
PROJECT(openvdbPrint)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

FILE(GLOB OPENVDBVIEWER_APP_HDRS_FILES cmd/openvdb_print/*.h ) 
FILE(GLOB OPENVDBVIEWER_SRCS_FILES cmd/openvdb_print/*.cc)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/..)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_EXECUTABLE(${PROJECT_NAME}
	${OPENVDBVIEWER_APP_HDRS_FILES}
	${OPENVDBVIEWER_SRCS_FILES}
  ${PROJECT_BINARY_DIR}/version.rc
)
ADD_DEPENDENCIES(${PROJECT_NAME} Core Viewer)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Boost_LIBRARIES} ${ZLIB_LIBRARY} ${OPENEXRHALF_LIBRARY} ${TBB_LIBRARY} openvdbCore)

# /bigobj is required for windows builds because of the size of
# some object files (tree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "/bigobj")
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES VERSION ${OPENVDB_VERSION})
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES FOLDER apps)

############################## Install
INSTALL(TARGETS ${PROJECT_NAME} 
	RUNTIME DESTINATION bin COMPONENT Runtime
	LIBRARY DESTINATION lib COMPONENT Development
	ARCHIVE DESTINATION lib COMPONENT Development)
	
#copy header files to include directory
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include COMPONENT Development FILES_MATCHING PATTERN "*.h" )
