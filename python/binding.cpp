/*
 * binding.cpp
 *
 *  Created on: Dec 24, 2020
 *      Author: sujiwo
 */

#include <iostream>
#include <vector>
#include "IncrementalBoW.h"
#include "VLAD.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sstream>
#include "cv_conversion.h"

#include <opencv2/core.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

//using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

using cVLAD=PlaceRecognizer::VLAD;
using cVisDict=PlaceRecognizer::VisualDictionary;


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

	void addImage(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints, uint placeId=-1)
	{
		if (bow.numImages()!=0)
			return addImage2(descriptors, keypoints);
		bow.addImage(cImageId, keypoints, descriptors);
		cImageId++;
	}

	void addImage2(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints)
	{
		bow.addImage2(cImageId, keypoints, descriptors);
		cImageId++;
	}

	std::vector<int> query(cv::Mat &descriptors, const uint numToReturn)
	{
		std::vector<std::vector<cv::DMatch>> descMatches;
		bow.searchDescriptors(descriptors, descMatches, 2, 32);

		// Filter matches according to ratio test
		std::vector<cv::DMatch> realMatches;
		for (uint m=0; m<descMatches.size(); m++) {
			if (descMatches[m][0].distance < descMatches[m][1].distance * 0.65)
				realMatches.push_back(descMatches[m][0]);
		}

		std::vector<PlaceRecognizer::ImageMatch> imageMatches, ret;
		bow.searchImages(descriptors, realMatches, imageMatches);

		ret = {imageMatches.begin(), imageMatches.begin() + std::min(numToReturn, (const uint)imageMatches.size())};

		std::vector<int> imgIds;
		for (auto &r: ret) {
			imgIds.push_back((int)r.image_id);
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
		cImageId = bow.numImages();
		return true;
	}

	uint numImages() const
	{ return bow.numImages(); }

	uint numDescriptors() const
	{ return bow.numDescriptors(); }

	uint lastImageId() const
	{ return numImages(); }

protected:
	PlaceRecognizer::IncrementalBoW bow;
	uint cImageId = 0;
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
				"descriptors"_a,
				"keypoints"_a,
				"placeId"_a=-1,
				"Add new image descriptors from an image")
			.def("stopTrain", &xIBoW::stopTrain,
				"End a training session")
			.def("query", &xIBoW::query, "descriptors"_a, "numOfImages"_a=5)

			.def("save", &xIBoW::save, "save mapped images to disk file")
			.def("load", &xIBoW::load, "Load a map file from disk file")

			.def_property_readonly("numImages", &xIBoW::numImages, "Number of images stored in database")
			.def_property_readonly("numDescriptors", &xIBoW::numDescriptors, "Number of descriptors stored in database")

			.def("lastImageId", &xIBoW::lastImageId)
		;

	py::class_<cVisDict> (mod, "cVisualDictionary")
		.def( py::init<>() )
		.def("setCenters", &cVisDict::setCenters)
		.def("getCenters", &cVisDict::getCenters)
		.def("predict", &cVisDict::predict, "Doc" )
	;

	py::class_<PlaceRecognizer::VLAD> (mod, "VLAD")
		.def( py::init<>() )
		.def( "initClusterCenters", &cVLAD::initClusterCenters,
			"Initialize dictionary's cluster centers using preloaded matrix")
		.def( "initTrain", &cVLAD::initTrain,
			"Initialize training session")
		.def( "addImage", &cVLAD::addImage,
			"descriptors"_a,
			"keypoints"_a,
			"placeId"_a=-1,
			"Add new image descriptors from an image")
		.def("stopTrain", &cVLAD::stopTrain,
			"End a training session")
		.def("query", &cVLAD::query, "descriptors"_a, "numOfImages"_a=5)

		.def("save", &cVLAD::save, "save mapped images to disk file")
		.def("load", &cVLAD::load, "Load a map file from disk file")

		.def("lastImageId", &cVLAD::lastImageId)
	;

}

