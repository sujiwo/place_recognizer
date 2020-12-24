#find_package (PythonLibs REQUIRED)
#find_package (Boost COMPONENTS python numpy REQUIRED)
#add_library(_place_recognizer MODULE binding.cpp )
#set_target_properties (_place_recognizer PROPERTIES 
#	PREFIX ""
#	LIBRARY_OUTPUT_NAME "_place_recognizer"
#	LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION} )
#target_include_directories(_place_recognizer PUBLIC 
#	${PROJECT_SOURCE_DIR}/include 
#	${PYTHON_INCLUDE_DIRS})
#target_link_libraries ( _place_recognizer
#	${Boost_LIBRARIES} 
#	${PYTHON_LIBRARIES}
#	${OpenCV_LIBS}
#	place_recognizer 
#	Boost::serialization
#	Boost::system
#	Boost::python
#	Boost::numpy)
#install(TARGETS _place_recognizer
#	LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION} )




set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11_catkin REQUIRED)
add_library ( py_place_recognizer
	MODULE
		binding2.cpp
		cv_conversion.cpp )
target_include_directories( py_place_recognizer
	BEFORE
	PRIVATE 
		${pybind11_catkin_INCLUDE_DIRS} )
set_target_properties (py_place_recognizer PROPERTIES 
	PREFIX ""
	LIBRARY_OUTPUT_NAME "_place_recognizer"
	LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION} )
target_link_libraries(py_place_recognizer
	PRIVATE
		place_recognizer
		${PYTHON_LIBRARIES}
		${OpenCV_LIBS}
)
install(
	TARGETS py_place_recognizer
		LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)



#install (FILES __init__.py VLAD.py
#	DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION} )

catkin_install_python(
	PROGRAMS
		__init__.py
		VLAD.py
	DESTINATION
		${CATKIN_PACKAGE_PYTHON_DESTINATION} )

# Python scripts for VLAD training, querying and service
catkin_install_python(
	PROGRAMS
		examples/cityscape_dict.py
		examples/query_from_bag.py
		examples/train_from_bag.py
		examples/server
	DESTINATION
		${CATKIN_PACKAGE_BIN_DESTINATION} )
		
		
		
		
