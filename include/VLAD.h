/*
 * VLAD.h
 *
 *  Created on: Nov 18, 2020
 *      Author: sujiwo
 */

#ifndef PLACE_RECOGNIZER_VLAD_H_
#define PLACE_RECOGNIZER_VLAD_H_

#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include "IncrementalBoW.h"


namespace PlaceRecognizer {


struct VisualDictionary
{
	VisualDictionary(uint _numWords=256) :
		numWords(_numWords)
	{}

	VisualDictionary(const cv::Mat& precomputedCenters);

	bool build (cv::Mat &descriptors);

	/*
	 * Returns a list of centroids for a set of descriptors
	 */
	std::vector<uint>
	predict(const cv::Mat &imageDescriptors) const;

	void
	predict(const cv::Mat &imageDescriptors, cv::Mat &out) const;

	/*
	 * Returns descriptors for the centroids
	 */
	inline cv::Mat getCenters() const
	{ return centers; }

protected:
	uint numWords;
	cv::Mat centers;

	uint predict1row(const cv::Mat &descriptor) const;
};


class VLAD {
public:
	VLAD(uint numWords=256, uint leafSize=40);
	virtual ~VLAD();

	void initTrain();

	void addImage(uint imageId, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);

	void stopTrain();

	bool search(const cv::Mat &image_descriptors, std::vector<ImageMatch> &results) const;

	bool save(const std::string &path);
	bool load(const std::string &path);

protected:
	uint leafSize;
	VisualDictionary vDict;
	std::vector<uint> imageIds;

	// training data structures
	std::vector<cv::Mat> trainDescriptors;
	std::map<uint, std::vector<uint>> imageIdsToDescriptors;

	cv::Mat computeVlad(const cv::Mat &descriptors) const;
};

} /* namespace PlaceRecognizer */

#endif /* PLACE_RECOGNIZER_VLAD_H_ */
