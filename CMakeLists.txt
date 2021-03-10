cmake_minimum_required(VERSION 3.0)
project(place_recognizer)

add_compile_options(-std=c++11 -frtti -fopenmp -msse4.2 -DOPENCV_TRAITS_ENABLE_DEPRECATED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	std_msgs
	cv_bridge
	sensor_msgs
	geometry_msgs
	message_generation
)

# See if we can compile mapper executable
find_package(rosbag_viewer) 

find_package (Boost REQUIRED COMPONENTS system serialization filesystem program_options)
find_package (OpenCV REQUIRED)
find_package (OpenMP REQUIRED)
find_package (Eigen3 REQUIRED)

if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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
	${EIGEN3_INCLUDE_DIRS}
)

add_library(
	place_recognizer
		lib/IncrementalBoW.cpp
		lib/VLAD.cpp
		lib/kdtree.cpp
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

install(
	TARGETS place_recognizer
		LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
		RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION} )

add_executable(bkmeans_test
	nodes/bkmeans_test.cpp)
target_link_libraries(bkmeans_test
	place_recognizer)


