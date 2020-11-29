/*
 * VLAD.h
 *
 *  Created on: Nov 18, 2020
 *      Author: sujiwo
 */

#ifndef PLACE_RECOGNIZER_VLAD_H_
#define PLACE_RECOGNIZER_VLAD_H_

#include <vector>
#include <opencv2/core.hpp>
#include "IncrementalBoW.h"


namespace PlaceRecognizer {


struct VisualDictionary
{
	VisualDictionary(uint _numWords=256) :
		numWords(_numWords)
	{}

	bool build (cv::Mat &descriptors);

	bool rebuild (cv::Mat &additional_descriptors);

	std::vector<uint>
	predict(const cv::Mat &imageDescriptors) const;

	cv::Mat getCenters() const
	{ return centers.clone(); }

protected:
	uint numWords;
	cv::Mat centers;

	uint predict1row(const cv::Mat &descriptor) const;
};


class VLAD {
public:
	VLAD();
	virtual ~VLAD();

	void initTrain();

	void train(uint imageId, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);

	void stopTrain();

	bool search(const cv::Mat &image_descriptors, std::vector<ImageMatch> &results) const;

protected:
	VisualDictionary vDict;
};

} /* namespace PlaceRecognizer */

#endif /* PLACE_RECOGNIZER_VLAD_H_ */
