cmake_minimum_required(VERSION 3.0)
project(place_recognizer)

add_compile_options(-std=c++11 -frtti -fopenmp -msse4.2)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(UseCython)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	std_msgs
	cv_bridge
	sensor_msgs
	message_generation
)

# See if we can compile mapper executable
find_package(rosbag_viewer) 

find_package (Boost REQUIRED COMPONENTS system serialization filesystem program_options)
find_package (OpenCV REQUIRED)
find_package (OpenMP REQUIRED)

if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(SEGNET_DIR)
	message("SegNet is set")
	include("${SEGNET_DIR}/share/Caffe/CaffeConfig.cmake")
	message("SegNet include: " "${Caffe_INCLUDE_DIRS}")
	set(SEGNET_FOUND true)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDNN -DUSE_LEVELDB -DUSE_LMDB -DUSE_OPENCV -DSEGNET_FOUND=1")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${SEGNET_DIR}/lib")
	set(SEGNET_WRAPPER "lib/Segmentation.cpp")
endif()

add_service_files(
	FILES
		place_recognizer.srv
)

generate_messages(
	DEPENDENCIES
		sensor_msgs
		std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES place_recognizer
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs cv_bridge
)

include_directories(
	include
	${CMAKE_CURRENT_SOURCE_DIR}
	${catkin_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

add_library(
	place_recognizer
		lib/IncrementalBoW.cpp
		lib/VLAD.cpp
		lib/BOWKmajorityTrainer.cpp
		lib/BKMeans.cpp
		lib/pam.cpp
		lib/KMedoids.cpp
)
target_link_libraries(
	place_recognizer
		${OpenCV_LIBS}
		${catkin_LIBRARIES}
		Boost::serialization
		Boost::system
)

# Python extension
add_subdirectory (python)

add_executable( recognizer_srv
	nodes/recognizer_srv.cpp
	${SEGNET_WRAPPER} )
target_link_libraries( recognizer_srv
	place_recognizer
	Boost::system
	Boost::program_options
	${catkin_LIBRARIES}
	${Caffe_LIBRARIES} )
target_include_directories(
	recognizer_srv
		PUBLIC
			${Caffe_INCLUDE_DIRS}
	)

install(
	TARGETS place_recognizer recognizer_srv
		LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
		RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION} )

add_executable(bkmeans_test
	nodes/bkmeans_test.cpp)
target_link_libraries(bkmeans_test
	place_recognizer)


if(rosbag_viewer_FOUND)
	message(STATUS "rosbag_viewer found; will compile mapper executables")
	find_package(sensor_msgs REQUIRED)
	find_package(cv_bridge REQUIRED)
	find_package(image_transport REQUIRED)
	
	add_executable(ibow_mapper 
		nodes/ibow_mapper.cpp
		lib/ImageBag.cpp
		${SEGNET_WRAPPER})
	target_include_directories(
		ibow_mapper 
			PUBLIC 
			${rosbag_viewer_INCLUDE_DIRS}
			${Caffe_INCLUDE_DIRS})
	target_link_libraries(
		ibow_mapper
			place_recognizer
			random_access_bag
			${Caffe_LIBRARIES}
			${image_transport_LIBRARIES})
	set_target_properties(
		ibow_mapper PROPERTIES
			COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDNN -DUSE_LEVELDB -DUSE_LMDB -DUSE_OPENCV -DSEGNET_FOUND=1"
			LINK_FLAGS "-L${SEGNET_DIR}/lib")
	install(
		TARGETS ibow_mapper
			RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif()
