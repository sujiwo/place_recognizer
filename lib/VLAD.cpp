/*
 * VLAD.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: sujiwo
 */

#include <algorithm>
#include "VLAD.h"
#include "BOWKmajorityTrainer.h"


using namespace std;


namespace PlaceRecognizer {

bool
VisualDictionary::build (cv::Mat &descriptors)
{
	cv::BOWKmajorityTrainer trainer(numWords);
	centers = trainer.cluster(descriptors);
	return (centers.rows==numWords);
}

bool
VisualDictionary::rebuild (cv::Mat &add_descriptors)
{
	// XXX: Not implemented
	return false;
}

std::vector<uint>
VisualDictionary::predict (const cv::Mat &imageDescriptors)
const
{
	vector<uint> prd(imageDescriptors.rows);
	for (int i=0; i<imageDescriptors.rows; ++i) {
		auto c = predict1row(imageDescriptors.row(i));
		prd[i] = c;
	}
	return prd;
}

uint
VisualDictionary::predict1row(const cv::Mat &descriptor) const
{
	assert(descriptor.type()==CV_8UC1);
	assert(descriptor.rows==1 and descriptor.cols==centers.cols);
	vector<uint> distances(numWords);
	for (int i=0; i<numWords; i++) {
		distances[i] = cv::norm(descriptor, centers.row(i), cv::NormTypes::NORM_HAMMING);
	}
	return min_element(distances.begin(), distances.end()) - distances.begin();
}


VLAD::VLAD(uint numWords, uint _leafSize) :
	vDict(numWords),
	leafSize(_leafSize)
{}

VLAD::~VLAD() {
	// TODO Auto-generated destructor stub
}

cv::Mat
VLAD::computeVlad(const cv::Mat &descriptors) const
{
	auto predictedLabels = vDict.predict(descriptors);
	// Unfinished
}

void
VLAD::initTrain()
{
	trainDescriptors.clear();
}

void
VLAD::addImage(uint imageId, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors)
{
	imageIds.push_back(imageId);
	// Unfinished
}

} /* namespace PlaceRecognizer */
