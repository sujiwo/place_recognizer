/*
 * VLAD.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: sujiwo
 */


#include <algorithm>
#include "VLAD.h"


using namespace std;


namespace PlaceRecognizer {

bool
VisualDictionary::build (cv::Mat &descriptors)
{
	auto term = cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 10, 0.1);
	cv::Mat bestLabels;
	auto d = cv::kmeans(
		descriptors,
		numWords,
		bestLabels,
		term,
		5,
		cv::KMEANS_PP_CENTERS,
		centers);
	return true;
}


std::vector<uint>
VisualDictionary::predict (const cv::Mat &imageDescriptors)
const
{
/*
	vector<uint> prd(imageDescriptors.rows);
	for (int i=0; i<imageDescriptors.rows; ++i) {
		auto c = predict1row(imageDescriptors.row(i));
		prd[i] = c;
	}
	return prd;
*/
}

uint
VisualDictionary::predict1row(const cv::Mat &descriptor) const
{
/*
	assert(descriptor.type()==CV_8UC1);
	assert(descriptor.rows==1 and descriptor.cols==centers.cols);
	vector<uint> distances(numWords);
	for (int i=0; i<numWords; i++) {
		distances[i] = cv::norm(descriptor, centers.row(i), cv::NormTypes::NORM_HAMMING);
	}
	return min_element(distances.begin(), distances.end()) - distances.begin();
*/
	assert(descriptor.rows==1 && descriptor.cols==centers.cols);

	cv::Mat descfloat;
	if (descriptor.type()!=CV_32FC1)
	descfloat = descriptor.convertTo(descfloat, CV_32FC1);
	else descfloat = descriptor;


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
