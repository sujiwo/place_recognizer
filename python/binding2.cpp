/*
 * binding2.cpp
 *
 *  Created on: Dec 24, 2020
 *      Author: sujiwo
 */

#include <iostream>
#include <vector>
#include "IncrementalBoW.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cv_conversion.h"

#include <opencv2/core.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;


void module_init()
{
	_import_array();
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

	void initTrain(int leafSize=40)
	{ /* do nothing */ }

	void addImage(const uint image_id, const cv::Mat &descriptors, const vector<cv::KeyPoint> &keypoints)
	{
		if (bow.numImages()!=0)
			return addImage2(image_id, descriptors, keypoints);
		return bow.addImage(image_id, keypoints, descriptors);
	}

	void addImage2(const uint image_id, const cv::Mat &descriptors, const vector<cv::KeyPoint> &keypoints)
	{
		return bow.addImage2(image_id, keypoints, descriptors);
	}

	vector<uint> query(cv::Mat &descriptors, const uint numToReturn)
	{
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

		vector<uint> imgIds;
		for (auto &r: ret) {
			imgIds.push_back((uint)r.image_id);
		}

		return imgIds;
	}

	void stopTrain()
	{ /* do nothing */ }

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



PYBIND11_MODULE(_place_recognizer, mod) {

	module_init();

	mod.doc() = "Python module for vision-based topological mapping and localization";

	py::class_<PlaceRecognizer::ImageMatch> (mod, "ImageMatch")
			.def_readonly("image_id", &PlaceRecognizer::ImageMatch::image_id)
			.def_readonly("score", &PlaceRecognizer::ImageMatch::score)
		;

	// Incremental BoW class
	py::class_<xIBoW> (mod, "IncrementalBoW")
			.def( py::init<>() )
			.def("initTrain", &xIBoW::initTrain,
				"Initialize training session", "leafSize"_a=40)
			.def("addImage", &xIBoW::addImage,
				"Add new image descriptors from an image")
			.def("stopTrain", &xIBoW::stopTrain,
				"End a training session")
			.def("query", &xIBoW::query, "descriptors"_a, "numOfImages"_a=5)
			.def("save", &xIBoW::save)
			.def("load", &xIBoW::load)
			.def_property_readonly("numImages", &xIBoW::numImages, "Number of images stored in database")
			.def_property_readonly("numDescriptors", &xIBoW::numDescriptors, "Number of descriptors stored in database")
		;

}
