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
namespace np=boost::python::numpy;
using namespace std;

/*
 * Helper function to convert from NDArray to OpenCV Mat
 */
cv::Mat convertNdArray(np::ndarray &A)
{
	auto ndims = A.get_nd();
	auto shape = A.get_shape();
	auto dtype = A.get_dtype();
	auto strides = A.get_strides();

	cv::Mat M;

	bool needcopy = false;

	const int rows = shape[0],
		cols = ndims>=2 ? shape[1] : 1,
		channel = ndims==3 ? shape[2] : 1;

	char* st = extract<char*>(str(dtype));
	int type;
	if (strcmp(st,"float64")==0)       type=CV_64F;
	else if (strcmp(st, "float32")==0) type=CV_32F;
	else if (strcmp(st, "int32")==0)   type=CV_32S;
	else if (strcmp(st, "int16")==0)   type=CV_16S;
	else if (strcmp(st, "uint16")==0)  type=CV_16U;
	else if (strcmp(st, "int8")==0)    type=CV_8S;
	else if (strcmp(st, "uint8")==0)   type=CV_8U;
	else throw runtime_error("Unsupported datatype");

	int elemsize = CV_ELEM_SIZE1(type),
	    cvtype = CV_MAKETYPE(type, channel);
	bool ismultichannel = ndims == 3 && shape[2] <= CV_CN_MAX;

	for( int i = ndims-1; i >= 0 && !needcopy; i-- )
	{
		// these checks handle cases of
		//  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
		//  b) transposed arrays, where _strides[] elements go in non-descending order
		//  c) flipped arrays, where some of _strides[] elements are negative
		// the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
		if( (i == ndims-1 && shape[i] > 1 && (size_t)strides[i] != elemsize) ||
				(i < ndims-1 && shape[i] > 1 && strides[i] < strides[i+1]) )
			needcopy = true;
	}
    if( ismultichannel && strides[1] != (npy_intp)elemsize*shape[2] )
        needcopy = true;

    if (needcopy) {
    	// Arrgh! not handled
    }
    else {
    	M = cv::Mat(rows, cols, cvtype, A.get_data());
    }

	return M;
}


void acceptString(const string &s)
{
	cout << s << endl;
}


void acceptKeypoint(object _k)
{
	cv::KeyPoint k;
	pyopencv_to(_k.ptr(), k);
	cout << k.pt << endl;
}


void acceptMat(np::ndarray _M)
{
	cv::Mat M = convertNdArray(_M);
	cout << M << endl;
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
	import_array();
	np::initialize();

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
