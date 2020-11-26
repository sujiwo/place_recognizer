find_package (PythonLibs REQUIRED)
find_package (Boost COMPONENTS python REQUIRED)

#set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(_place_recognizer MODULE binding.cpp conversion.cpp)
set_target_properties (_place_recognizer PROPERTIES 
	PREFIX ""
	LIBRARY_OUTPUT_NAME "_place_recognizer"
	LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION} )
target_include_directories(_place_recognizer PUBLIC 
	${PROJECT_SOURCE_DIR}/include 
	${PYTHON_INCLUDE_DIRS})
target_link_libraries ( _place_recognizer
	${Boost_LIBRARIES} 
	${PYTHON_LIBRARIES}
	${OpenCV_LIBS}
	place_recognizer 
	Boost::serialization
	Boost::system)
	
install (FILES __init__.py
	DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION} )

install(TARGETS _place_recognizer
	LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION} )
	