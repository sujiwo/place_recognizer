set_source_files_properties(
#	VLAD.pyx 
	iBoW.pyx 
PROPERTIES CYTHON_IS_CXX TRUE)

#cython_add_module(VLAD VLAD.pyx)
cython_add_module(iBoW iBoW.pyx)
target_include_directories(iBoW
	PUBLIC
		include
		${CMAKE_CURRENT_SOURCE_DIR}
		${PYTHON_INCLUDE_DIRS})

#set_target_properties(VLAD PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})
set_target_properties(iBoW PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

target_link_libraries (iBoW 	
	place_recognizer
	${OpenCV_LIBS} )

install (FILES __init__.py
	DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION} )

install(TARGETS iBoW
	LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
 