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


void
VLADDescriptor::compute(const cv::Mat &imageDescriptors, const VisualDictionary &dict)
{
	// XXX: Unfinished
	cv::Mat predictedLabels;
	dict.predict(imageDescriptors, predictedLabels);
	auto centers = dict.getCenters();
	int k = centers.rows,
		m = imageDescriptors.rows,
		d = imageDescriptors.cols;
	descriptors = cv::Mat::zeros(k, d, CV_32FC1);
	centroid_counters = vector<uint>(k, 0);

	for (auto r=0; r<imageDescriptors.rows; ++r) {
		auto descRow = imageDescriptors.row(r);
		auto label = predictedLabels.at<int>(0, r);
		centroid_counters[label] += 1;
		descriptors.row(label) += (descRow - centers.row(label));
	}
/*
	for (int i=0; i<k; ++k) {
		centroid_counters[i] = cv::countNonZero(predictedLabels==i);
		if (centroid_counters[i] > 0) {
		}
	}
*/
}


void
VLADDescriptor::adaptNewCentroids(const VisualDictionary &newDict, const cv::Mat &oldCentroids)
{
	for (int i=0; i<descriptors.rows; ++i) {
		descriptors.row(i) += centroid_counters[i]*(newDict.centers.row(i) - oldCentroids.row(i));
	}
}


cv::Mat
VLADDescriptor::normalized() const
{
	auto V = descriptors.clone();
	for (int r=0; r<descriptors.rows; ++r) {
		auto l2 = cv::norm(descriptors.row(r));
		if (l2 < 1e-3)
			V.row(r).setTo(0);
		else {
			V.row(r) /= l2;
		}
	}
	V /= cv::norm(V);
	return V;
}


cv::Mat
VLADDescriptor::flattened() const
{
	return normalized().reshape(0, 1);
}


void
VisualDictionary::setCenters(const cv::Mat& precomputedCenters)
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
	// XXX: Parallelize this
	for (int i=0; i<imageDescriptors.rows; ++i) {
		auto c = predict1row(imageDescriptors.row(i));
		prd[i] = c;
	}
	return prd;
}

void
VisualDictionary::predict(const cv::Mat &imageDescriptors, cv::Mat &prd) const
{
	prd = cv::Mat::zeros(1, imageDescriptors.rows, CV_32SC1);
	for (int i=0; i<imageDescriptors.rows; ++i) {
		auto c = predict1row(imageDescriptors.row(i));
		prd.at<int>(0,i) = c;
	}
}


cv::Mat
VisualDictionary::adapt(cv::InputArray newDescriptors, bool dryRun)
{
	auto newDescMat = newDescriptors.getMat();
	auto descCenters = predict(newDescMat);
	cv::Mat movingAverage = cv::Mat::zeros(centers.size(), CV_64FC1);
	vector<uint64_t> descCount(numWords, 0);

	// XXX: Parallelize this
	for (int i=0; i<newDescMat.rows; ++i) {
		auto c = descCenters[i];
		descCount[c] += 1;
		movingAverage.row(c) += newDescMat.row(i);
	}
	for (int i=0; i<numWords; ++i) {
		movingAverage.row(i) /= float(descCount[i]);
	}

	movingAverage = (movingAverage + centers) / 2.0;
	if (dryRun)
		return movingAverage;
	else centers = movingAverage.clone();
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
	trainDescriptorPtr.clear();
}

void
VLAD::addImage(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors, uint imageId)
{
	imageIds.push_back(imageId);

	auto curPtr = trainDescriptors.size();
	trainDescriptorPtr.push_back(make_pair(curPtr, curPtr+descriptors.rows));
	for (uint r=0; r<descriptors.rows; r++) {
		trainDescriptors.push_back(descriptors.row(r).clone());
	}
}


void
VLAD::rebuildVladDescriptors(const cv::Mat &oldDict)
{
	// XXX: Parallelize this
	for (auto &v: vDescriptors) {
		v.adaptNewCentroids(vDict, oldDict);
	}
}


cv::Mat
VLAD::flatNormalDescriptors() const
{
	auto shape = vDescriptors[0].descriptors.size;
	auto type = vDescriptors[0].descriptors.type();
	cv::Mat flatDesc = cv::Mat::zeros(vDescriptors.size(), shape[0]*shape[1], type);

	for (int i=0; i<vDescriptors.size(); ++i) {
		vDescriptors[i].flattened().copyTo(flatDesc.row(i));
	}
	return flatDesc;
}


void
VLAD::stopTrain()
{
	auto hasTrained = !(vDescriptors.size()==0);

	cout << "Cluster center adaptation\n";
	cv::Mat oldDict = vDict.centers.clone();
	vDict.adapt(trainDescriptors);

	if (hasTrained) {
		cout << "Adapting old VLAD descriptors to new centroids\n";
		rebuildVladDescriptors(oldDict);
	}

	cout << "Build VLAD from data stream\n";
	for (auto p=0; p<trainDescriptorPtr.size(); ++p) {
		auto trDescPtr = trainDescriptorPtr[p];
		vector<cv::Mat> trVecDesc (trainDescriptors.begin()+trDescPtr.first, trainDescriptors.begin()+trDescPtr.second);
		VLADDescriptor newvd(trVecDesc, vDict);
		vDescriptors.push_back(newvd);
	}

	auto D = flatNormalDescriptors();
	// XXX: find KDTree solution
}


} /* namespace PlaceRecognizer */
