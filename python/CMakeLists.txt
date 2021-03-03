set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11_catkin REQUIRED)
add_library ( py_place_recognizer
	MODULE
		binding.cpp
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



# Python modules w/ core classes 
# XXX: Should use catkin_install_python, but ROS Noetic
# mangles the installed files (not all) such that they are only
# `symlink' to source files but can't be imported
install(
	PROGRAMS
		__init__.py
		VLAD.py
		Segmentation.py
		GeographicCoordinate.py
		ImageSubscriber.py
		GenericImageMap.py
		MiscProcess.py
		ImageBagWithPose.py
		OxfordDataset.py
		OxfordDatasetViewer.py
	DESTINATION
		${CATKIN_PACKAGE_PYTHON_DESTINATION} )

# Python scripts for VLAD training, querying and service
install(
	PROGRAMS
		examples/cityscape_dict.py
		examples/query_from_bag.py
		examples/train_from_bag.py
		examples/server.py
		examples/pose_from_utm.py
		examples/train_from_ros.py
	DESTINATION
		${CATKIN_PACKAGE_BIN_DESTINATION} )
install(
	FILES
		examples/cityscape-cluster-centers.dat.npy
	DESTINATION
		${CATKIN_PACKAGE_SHARE_DESTINATION} )

		
		
		
