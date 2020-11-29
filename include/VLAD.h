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


namespace PlaceRecognizer {


struct VisualDictionary
{
	VisualDictionary(uint _numWords=256) :
		numWords(_numWords)
	{}

	bool build (cv::Mat &descriptors);

	bool rebuild (cv::Mat &additional_descriptors);

	cv::Mat predict(const cv::Mat &imageDescriptors);

protected:
	uint numWords;
	cv::Mat centers;
};


class VLAD {
public:
	VLAD();
	virtual ~VLAD();

	void initTrain();

	void train(uint imageId, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);

	void stopTrain();

protected:
};

} /* namespace PlaceRecognizer */

#endif /* PLACE_RECOGNIZER_VLAD_H_ */
