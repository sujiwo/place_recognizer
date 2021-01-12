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
#include "fstream_conversion.h"
#include "cv_conversion.h"

#include <opencv2/core.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

//using namespace std;
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

	std::vector<uint> query(cv::Mat &descriptors, const uint numToReturn)
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

		std::vector<uint> imgIds;
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

	bool save(BinaryStream &fd)
	{
		auto pfd = std::shared_ptr<std::ostream>(new std::ostream(&fd));
		bow.saveToDisk(*pfd);
		return true;
	}

	bool load(const std::string &path)
	{
		bow.loadFromDisk(path);
		cImageId = bow.numImages();
		return true;
	}

	bool load(BinaryStream &fd)
	{
		auto pfd = std::shared_ptr<std::istream>(new std::istream(&fd));
		bow.loadFromDisk(*pfd);
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

			.def("save", static_cast<bool (xIBoW::*)(const std::string&)>(&xIBoW::save), "Save mapped images to disk")
			.def("save", static_cast<bool (xIBoW::*)(BinaryStream&)>(&xIBoW::save), "Save mapped images to an open file descriptor")

			.def("load", static_cast<bool (xIBoW::*)(const std::string&)>(&xIBoW::load), "Load a map file from disk")
			.def("load", static_cast<bool (xIBoW::*)(BinaryStream&)>(&xIBoW::load), "Load a map file from an open file descriptor")

			.def_property_readonly("numImages", &xIBoW::numImages, "Number of images stored in database")
			.def_property_readonly("numDescriptors", &xIBoW::numDescriptors, "Number of descriptors stored in database")

			.def("lastImageId", &xIBoW::lastImageId);
		;

	// Experimental function to test Python<->C++ file handler
	mod.def("handle_write",
			[](BinaryStream &bfd)
			{
				auto pfd = std::shared_ptr<std::ostream>(new std::ostream(&bfd));
				*pfd << "Binary";
			},
			py::arg("fd"),
			"File handler sample"
		);

	mod.def("handle_read",
			[](BinaryStream &bfd)
			{
				auto pfd = std::shared_ptr<std::istream>(new std::istream(&bfd));
			}
		);
}

