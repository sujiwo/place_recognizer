/*
 * VLAD.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: sujiwo
 */


#include <vector>
#include <algorithm>
#include <numeric>
#include <opencv2/core/persistence.hpp>
#include "VLAD.h"


using namespace std;


namespace PlaceRecognizer {


void
VLADDescriptor::compute(const cv::Mat &imageDescriptors, const VisualDictionary &dict)
{
	// XXX: Unfinished
	auto predictedLabels = dict.predict(imageDescriptors);

	auto centers = dict.getCenters();
	int k = centers.rows,
		m = imageDescriptors.rows,
		d = imageDescriptors.cols;
	descriptors = cv::Mat::zeros(k, d, CV_32FC1);
	centroid_counters = vector<uint>(k, 0);

	for (auto r=0; r<imageDescriptors.rows; ++r) {
		auto descRow = imageDescriptors.row(r);
		auto label = predictedLabels[r];
		centroid_counters[label] += 1;
		descriptors.row(label) += (descRow - centers.row(label));
	}
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

	cv::Mat descfloat;
	if (imageDescriptors.type()!=CV_32FC1)
		imageDescriptors.convertTo(descfloat, CV_32FC1);
	else descfloat = imageDescriptors;

#pragma omp parallel for
	for (int r=0; r<imageDescriptors.rows; ++r) {
		auto c = predict1row(descfloat, r);
		prd[r] = c;
	}
	return prd;
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
VisualDictionary::predict1row(const cv::Mat &descriptors, int rowNum) const
{
/*
	vector<float> norms2(numWords);
	for (int i=0; i<numWords; ++i)
		norms2[i] = cv::norm(centers.row(i) - descriptors.row(rowNum));
	return min_element(norms2.begin(), norms2.end()) - norms2.begin();
*/
	cv::Mat tmp(numWords, descriptors.cols, CV_32F),
		sum(numWords, 1, CV_32F);

	cv::repeat(descriptors.row(rowNum), numWords, 1, tmp);
	tmp -= centers;
	tmp = tmp.mul(tmp);
	cv::reduce(tmp, sum, 1, cv::REDUCE_SUM);
	cv::sqrt(sum, sum);
	int minloc;
	cv::minMaxIdx(sum, 0, 0, &minloc, NULL);
	return minloc;
}


VLAD::VLAD(uint numWords, uint _leafSize) :
	vDict(numWords),
	leafSize(_leafSize),
	searchTree(cv::ml::KNearest::create())
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
	searchTree->clear();
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


std::vector<uint>
VLAD::query(const cv::Mat &descriptors, const uint numToReturn) const
{
	VLADDescriptor queryDesc(descriptors, vDict);
	auto vlad = queryDesc.flattened();

	vector<uint> results;
	auto resp = searchTree->predict(vlad, results);
}


bool
VLAD::save(const std::string &path)
{
	try {
		cv::FileStorage store(path, cv::FileStorage::Mode::WRITE);
		searchTree->write(store);
		store.write("cluster_centers", vDict.centers);
		store.release();
	} catch (exception &e) {
		return false;
	}

	return true;
}


bool
VLAD::load(const std::string &path)
{
	try {
		cv::FileStorage store(path, cv::FileStorage::Mode::READ);
		searchTree = cv::Algorithm::read<cv::ml::KNearest>(store.root());
		vDict.setCenters(store["cluster_centers"].mat());
	} catch (exception &e) {
		return false;
	}

	return true;
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
	searchTree->setAlgorithmType(cv::ml::KNearest::Types::KDTREE);
	searchTree->setIsClassifier(true);

	cv::Mat_<int> responses(1, D.rows);
	std::iota(responses.begin(), responses.end(), 0);

	searchTree->train(D, cv::ml::ROW_SAMPLE, responses);
}


} /* namespace PlaceRecognizer */
