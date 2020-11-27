/*
 * binding.cpp
 *
 *  Created on: Nov 26, 2020
 *      Author: sujiwo
 */

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "IncrementalBoW.h"
#include "BOWKmajorityTrainer.h"
#include "conversion.h"

using namespace boost::python;
using namespace std;


void acceptString(const string &s)
{
	cout << s << endl;
}


void acceptKeypoint(PyObject *_k)
{
	cv::KeyPoint k;
	pyopencv_to(_k, k);
	cout << k.pt << endl;
}

void acceptMat(PyObject *_M)
{
	cv::Mat M;
	pyopencv_to(_M, M);
	cout << M.rows << endl;
}

class xIBoW
{
public:
xIBoW()
{}

void addImage(const uint image_id, PyObject* _keypoints, PyObject* _descriptors)
{
/*
	vector<cv::KeyPoint> keypoints;
	pyopencv_to(_keypoints, keypoints);
	auto descriptors = converter.toMat(_descriptors);
	bow.addImage(image_id, keypoints, descriptors);
*/
}

protected:
	PlaceRecognizer::IncrementalBoW bow;
};


static void minit()
{
	Py_Initialize();
	numpy::initialize();
	import_array();
}

BOOST_PYTHON_MODULE(_place_recognizer)
{
	minit();

	def("acceptString", acceptString);
	def("acceptKeypoint", acceptKeypoint);
	def("acceptMat", acceptMat);

	class_<xIBoW>("IncrementalBoW")
		.def("addImage", &xIBoW::addImage)
	;
}
