/*
 * VLAD.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: sujiwo
 */


#include <vector>
#include <algorithm>
#include "VLAD.h"


using namespace std;


namespace PlaceRecognizer {


struct VLADDescriptor
{
	VLADDescriptor(const cv::Mat &imageDescriptors, const VisualDictionary &dict)
	{ compute(imageDescriptors, dict); }

	void compute(const cv::Mat &imageDescriptors, const VisualDictionary &dict)
	{
		cv::Mat predictedLabels;
		dict.predict(imageDescriptors, predictedLabels);
		auto centers = dict.getCenters();
		int k = centers.rows,
			m = imageDescriptors.rows,
			d = imageDescriptors.cols;
		descriptors = cv::Mat::zeros(k, d, CV_32FC1);
		centroid_counters.reserve(k);

		for (int i=0; i<k; ++k) {
			centroid_counters[i] = cv::countNonZero(predictedLabels==i);
			if (centroid_counters[i] > 0) {
//				descriptors
			}
		}


	}

	cv::Mat normalized() const
	{

	}

	cv::Mat flattened() const
	{

	}

	void adaptNewCentroids(const VisualDictionary &newDict, const cv::Mat &oldCentroids)
	{

	}

	cv::Mat descriptors;
	vector<uint> centroid_counters;
};


VisualDictionary::VisualDictionary(const cv::Mat& precomputedCenters)
{
	assert(precomputedCenters.type()==CV_32FC1);
	centers = precomputedCenters.clone();
	numWords = centers.rows;
}


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
	vector<uint> prd(imageDescriptors.rows);
	for (int i=0; i<imageDescriptors.rows; ++i) {
		auto c = predict1row(imageDescriptors.row(i));
		prd[i] = c;
	}
	return prd;
}

void
VisualDictionary::predict(const cv::Mat &imageDescriptors, cv::Mat &prd) const
{
	prd = cv::Mat(1, imageDescriptors.rows, CV_32SC1);
	for (int i=0; i<imageDescriptors.rows; ++i) {
		auto c = predict1row(imageDescriptors.row(i));
		prd.at<int>(0,i) = c;
	}
}


uint
VisualDictionary::predict1row(const cv::Mat &descriptor) const
{
	assert(descriptor.rows==1 && descriptor.cols==centers.cols);

	cv::Mat descfloat;
	if (descriptor.type()!=CV_32FC1)
		descriptor.convertTo(descfloat, CV_32FC1);
	else descfloat = descriptor;

	vector<double> norms2(numWords);
	for (int i=0; i<numWords; ++i)
		norms2[i] = cv::norm(centers.row(i) - descfloat);
	return min_element(norms2.begin(), norms2.end()) - norms2.begin();
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
