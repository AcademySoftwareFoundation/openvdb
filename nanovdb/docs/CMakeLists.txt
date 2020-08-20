
#find_package(doxygen REQUIRED dot  )
if (WIN32)
	SET (DOXYGEN_EXECUTABLE "C:/Program Files/doxygen/bin/doxygen.exe")
else()	
	SET (DOXYGEN_EXECUTABLE "doxygen")
endif()

MESSAGE ( STATUS "DOXYGEN_EXECUTABLE = ${DOXYGEN_EXECUTABLE}")

FILE ( READ ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-config DOXYGEN_CONFIG_CONTENT )
FILE ( WRITE ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config ${DOXYGEN_CONFIG_CONTENT} )
FILE ( APPEND ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config "OUTPUT_DIRECTORY=\"${CMAKE_CURRENT_BINARY_DIR}/docs\"\n" )

ADD_CUSTOM_TARGET ( nanovdb_docs
SOURCES doxygen-config doc.md codingstyle.txt ToDo.md FAQ.md SourceTree.md HowToBuild.md HelloWorld.md
COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config
WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)

INSTALL ( CODE "EXECUTE_PROCESS(COMMAND \"${CMAKE_COMMAND}\" --build  \"${PROJECT_BINARY_DIR}\" --target nanovdb_docs)")
INSTALL ( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/html DESTINATION docs)

