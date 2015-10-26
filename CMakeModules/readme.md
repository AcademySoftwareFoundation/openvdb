External libraries are discovered using the CMAKE syntax that uses CONFIG files.
https://cmake.org/cmake/help/v3.2/command/find_package.html?highlight=find_package

You will need to setup the external libraries in their respective folders such that both CMAKE_PREFIX_PATH contains a path to that folders that contains the two cmake configuration file [libName]Config.cmake and [libName]ConfigVersion.cmake.

You can find an example of such files in the folder /CMakeModules/configModules.

CMAKE must be told to look for these folders by properly setting the variable CMAKE_PREFIX_PATH 
