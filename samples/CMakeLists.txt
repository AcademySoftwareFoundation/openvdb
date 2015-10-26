# add a very simple test wig=th openvdb
add_executable(openVDBHelloWorld openVDBHelloWorld.cpp)
add_dependencies(openVDBHelloWorld openvdbCore)
set_target_properties(openVDBHelloWorld PROPERTIES VERSION ${PROJECT_VER} SOVERSION ${PROJECT_APIVER})	
set_target_properties(openVDBHelloWorld PROPERTIES FOLDER samples)
target_link_libraries(openVDBHelloWorld openvdbCore)

add_executable(openVDBLevelSetSphere openVDBLevelSetSphere.cpp)
add_dependencies(openVDBLevelSetSphere openvdbCore)
set_target_properties(openVDBLevelSetSphere PROPERTIES VERSION ${PROJECT_VER} SOVERSION ${PROJECT_APIVER})	
set_target_properties(openVDBLevelSetSphere PROPERTIES FOLDER samples)
target_link_libraries(openVDBLevelSetSphere openvdbCore)
