/*
 * binding.cpp
 *
 *  Created on: Nov 26, 2020
 *      Author: sujiwo
 */

#include <iostream>
#include <boost/python.hpp>
#include "conversion.h"
#include "IncrementalBoW.h"
#include "BOWKmajorityTrainer.h"

using namespace boost::python;
using namespace std;

class xIBoW
{
public:
xIBoW()
{}

void addImage(const uint image_id, PyObject* _keypoints, PyObject* _descriptors)
{
	vector<cv::KeyPoint> keypoints;
	pyopencv_to(_keypoints, keypoints);
	auto descriptors = converter.toMat(_descriptors);
	bow.addImage(image_id, keypoints, descriptors);
}

protected:
	PlaceRecognizer::IncrementalBoW bow;
	NDArrayConverter converter;
};


static void minit()
{
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(_place_recognizer)
{
	minit();
	class_<xIBoW>("IncrementalBoW")
		.def("addImage", &xIBoW::addImage)
	;
}
