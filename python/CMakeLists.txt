set_source_files_properties(
#	VLAD.pyx 
	_place_recognizer.pyx 
PROPERTIES CYTHON_IS_CXX TRUE)

#cython_add_module(VLAD VLAD.pyx)
cython_add_module(_place_recognizer _place_recognizer.pyx conversion.cpp)
target_include_directories(_place_recognizer
	PUBLIC
		include
		${CMAKE_CURRENT_SOURCE_DIR}
		${PYTHON_INCLUDE_DIRS})

#set_target_properties(VLAD PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})
set_target_properties(_place_recognizer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

target_link_libraries (_place_recognizer 	
	place_recognizer
	${OpenCV_LIBS} )

install (FILES __init__.py
	DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION} )

install(TARGETS _place_recognizer
	LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
 