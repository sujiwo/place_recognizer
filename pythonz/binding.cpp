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
#include "VLAD.h"


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

//#include "conversion.h"

using namespace boost::python;
namespace py=boost::python;
namespace np=boost::python::numpy;
using namespace std;


#define ListSize(l) ((PyListObject*)l.ptr())->ob_size

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
		throw runtime_error("Non-continuous array can't be handled yet");
	}
	else {
		M = cv::Mat(rows, cols, cvtype, A.get_data());
		M.addref();
	}

	return M;
}


cv::KeyPoint convertKeyPoint(object &O)
{
	assert(string(extract<string>(O.attr("__class__")))=="cv2.KeyPoint");
	cv::KeyPoint kp;

	kp.pt.x = extract<float>(O.attr("pt")[0]);
	kp.pt.y = extract<float>(O.attr("pt")[1]);
	kp.angle = extract<float>(O.attr("angle"));
	kp.class_id = extract<int>(O.attr("class_id"));
	kp.octave = extract<int>(O.attr("octave"));
	kp.response = extract<float>(O.attr("response"));
	kp.size = extract<float>(O.attr("size"));

	return kp;
}


vector<cv::KeyPoint> convertKeyPointList(py::list &L)
{
	vector<cv::KeyPoint> vKp;
	uint listsz=ListSize(L);
	for (uint i=0; i<listsz; i++) {
		object _k = extract<py::object>(L[i]);
		auto kp = convertKeyPoint(_k);
		vKp.push_back(kp);
	}

	return vKp;
}


cv::DMatch convertDMatch(object &O)
{
	assert(string(extract<string>(O.attr("__class__")))=="cv2.DMatch");
	cv::DMatch D;
	return D;
}


void acceptString(const string &s)
{
	cout << s << endl;
}


void acceptKeypoint(object &_k)
{
	cv::KeyPoint K = extract<cv::KeyPoint>(_k);
	cout << K.size << endl;
}


void acceptList (py::list &_LK)
{
	uint listSize = ListSize(_LK);
}


void acceptMat(np::ndarray _M)
{
	cv::Mat M = convertNdArray(_M);
	cout << M << endl;
}


class xIBoW
{
public:
	xIBoW(
			const uint k = 16,
			const uint s = 150,
			const uint t = 4,
			const PlaceRecognizer::IncrementalBoW::MergePolicy merge_policy =
					PlaceRecognizer::IncrementalBoW::MergePolicy::MERGE_POLICY_NONE,
			const bool purge_descriptors = true,
			const uint min_feat_apps = 3) :
		bow(k, s, t, merge_policy, min_feat_apps)
	{}

	void addImage(const uint image_id, py::list &_keypoints, np::ndarray &_descriptors)
	{
		if (bow.numImages()!=0)
			return addImage2(image_id, _keypoints, _descriptors);

		auto vKeys = convertKeyPointList(_keypoints);
		auto descriptors = convertNdArray(_descriptors);
		return bow.addImage(image_id, vKeys, descriptors);
	}

	void addImage2(const uint image_id, py::list &_keypoints, np::ndarray &_descriptors)
	{
		auto vKeys = convertKeyPointList(_keypoints);
		auto descriptors = convertNdArray(_descriptors);
		return bow.addImage2(image_id, vKeys, descriptors);
	}

	vector<PlaceRecognizer::ImageMatch> search(np::ndarray &_descriptors, const uint numToReturn)
	{
		auto descriptors = convertNdArray(_descriptors);

		vector<vector<cv::DMatch>> descMatches;
		bow.searchDescriptors(descriptors, descMatches, 2, 32);

		// Filter matches according to ratio test
		vector<cv::DMatch> realMatches;
		for (uint m=0; m<descMatches.size(); m++) {
			if (descMatches[m][0].distance < descMatches[m][1].distance * 0.65)
				realMatches.push_back(descMatches[m][0]);
		}

		vector<PlaceRecognizer::ImageMatch> imageMatches, ret;
		bow.searchImages(descriptors, realMatches, imageMatches);

		ret = {imageMatches.begin(), imageMatches.begin()+min(numToReturn, (const uint)imageMatches.size())};

		return ret;
	}

	bool save(const std::string &path)
	{
		bow.saveToDisk(path);
		return true;
	}

	bool load(const std::string &path)
	{
		bow.loadFromDisk(path);
		return true;
	}

	uint numImages() const
	{ return bow.numImages(); }

	uint numDescriptors() const
	{ return bow.numDescriptors(); }

protected:
	PlaceRecognizer::IncrementalBoW bow;
};


class xVLAD
{
public:
	xVLAD() {}

	void initTrain()
	{}

	void stopTrain()
	{}

	void train(uint imageId, py::list &_keypoints, np::ndarray &_descriptors)
	{}

protected:
	PlaceRecognizer::VLAD mvlad;
};


np::ndarray kmajority(np::ndarray &input)
{
	auto Inp = convertNdArray(input);
}


static void minit()
{
	Py_Initialize();
	import_array();
	np::initialize();

}


BOOST_PYTHON_MODULE(_place_recognizer)
{
	minit();

	/*
	 * Toy functions to test converter
	 */
	def("acceptString", acceptString);
	def("acceptKeypoint", acceptKeypoint);
	def("acceptMat", acceptMat);
	def("acceptList", acceptList);

	def("kmajority", kmajority);

	enum_<PlaceRecognizer::IncrementalBoW::MergePolicy>("MergePolicy")
		.value("MERGE_POLICY_NONE", PlaceRecognizer::IncrementalBoW::MergePolicy::MERGE_POLICY_NONE)
		.value("MERGE_POLICY_AND", PlaceRecognizer::IncrementalBoW::MergePolicy::MERGE_POLICY_AND)
		.value("MERGE_POLICY_OR", PlaceRecognizer::IncrementalBoW::MergePolicy::MERGE_POLICY_OR)
	;

	class_<PlaceRecognizer::ImageMatch>("ImageMatch")
		.def_readonly("image_id", &PlaceRecognizer::ImageMatch::image_id)
		.def_readonly("score", &PlaceRecognizer::ImageMatch::score)
	;

	class_<xIBoW>("IncrementalBoW",
			init<uint,uint,uint,PlaceRecognizer::IncrementalBoW::MergePolicy,bool,uint>
				(args("k", "s", "t", "merge_policy", "min_feat_apps")) )
		.def(init<>())
		.def("addImage", &xIBoW::addImage)
		.def("addImage2", &xIBoW::addImage2)
		.def("search", &xIBoW::search)
		.def("save", &xIBoW::save)
		.def("load", &xIBoW::load)
		.def_readonly("numImages", &xIBoW::numImages)
		.def_readonly("numDescriptors", &xIBoW::numDescriptors)
	;

	class_<xVLAD>("VLAD")
		// Add defs here
	;

//	boost::python::converter::registry::push_back(expected_pytype)
}

