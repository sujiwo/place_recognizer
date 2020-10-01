/*
 * Segmentation.h
 *
 *  Created on: Feb 22, 2020
 *      Author: sujiwo
 *
 * The objective is to create a mask at which no image features will not fall upon that areas
 */

#ifndef PLACE_RECOGNIZER_SEGMENTATION_H_
#define PLACE_RECOGNIZER_SEGMENTATION_H_


#include <string>
#include <memory>
#include <opencv2/core.hpp>
#define USE_OPENCV 1
#include <caffe/caffe.hpp>


namespace PlaceRecognizer {

class Segmentation
{
public:
	Segmentation(const std::string &modelPath, const std::string &weights);

	cv::Mat segment(const cv::Mat &sourceImage);

	cv::Mat buildMask(const cv::Mat &sourceImage);

	virtual ~Segmentation();

	/*
	 * Return last result
	 */
	inline cv::Mat getLastResult() const
	{ return lastSegnResult.clone(); }

protected:
	std::shared_ptr<caffe::Net<float>> mNet;
	cv::Size imgInputSize;
	uint numChannels;

	cv::Mat lastSegnResult;
};

} /* namespace PlaceRecognizer */

#endif /* PLACE_RECOGNIZER_SEGMENTATION_H_ */
