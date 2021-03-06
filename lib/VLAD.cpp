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
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "cvobj_serialization.h"
#include "VLAD.h"


using namespace std;


namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive &ar, cv::ml::KDTree::Node &node, const unsigned int version)
{
	ar & node.idx & node.left & node.right & node.boundary;
}

template<class Archive>
void serialize(Archive &ar, cv::ml::KDTree &tree, const unsigned int version)
{
	ar & tree.nodes;
	ar & tree.points;
	ar & tree.labels;
	ar & tree.maxDepth;
	ar & tree.normType;
}

}
}


namespace PlaceRecognizer {


cv::Mat createMatFromVectorMat(const std::vector<cv::Mat> &src)
{
	cv::Mat dst(src.size(), src[0].cols, src[0].type());
	for (int r=0; r<src.size(); ++r)
		src[r].copyTo(dst.row(r));
	return dst;
}


VLADDescriptor::VLADDescriptor(const cv::Mat &imageDescriptors, const VisualDictionary &dict)
{
	compute(imageDescriptors, dict);
}


VLADDescriptor::VLADDescriptor(const std::vector<cv::Mat> imageDescriptors, const VisualDictionary &dict)
{
	auto M = createMatFromVectorMat(imageDescriptors);
	compute(M, dict);
}



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
	centroid_counters = Counters(k, 0);

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
		cv::Mat R;
		newDescMat.row(i).convertTo(R, movingAverage.type());
		movingAverage.row(c) += R;
	}
	for (int i=0; i<numWords; ++i) {
		movingAverage.row(i) /= float(descCount[i]);
	}

	movingAverage.convertTo(movingAverage, centers.type());
	movingAverage = (movingAverage + centers) / 2.0;
	if (dryRun)
		return movingAverage;
	else {
		centers = movingAverage.clone();
		return centers;
	}
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
	// This version is faster than above
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
	leafSize(_leafSize)
{}

VLAD::~VLAD() {
	// TODO Auto-generated destructor stub
}

void
VLAD::initTrain()
{
	trainDescriptors.clear();
	trainDescriptorPtr.clear();
	kdtree = cv::ml::KDTree();
}

void
VLAD::addImage(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints, uint imageId)
{
	imageIds.push_back(imageId);

	cv::Mat descriptorsFloat;
	// make sure to get 32-bit float
	descriptors.convertTo(descriptorsFloat, CV_32F);

	auto curPtr = trainDescriptors.size();
	trainDescriptorPtr.push_back(make_pair(curPtr, curPtr+descriptorsFloat.rows));
	for (uint r=0; r<descriptorsFloat.rows; r++) {
		trainDescriptors.push_back(descriptorsFloat.row(r).clone());
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


std::vector<int>
VLAD::query(const cv::Mat &descriptors, const uint numToReturn) const
{
	cv::Mat descFloat;
	if (descriptors.type()!=CV_32F)
		descriptors.convertTo(descFloat, CV_32F);
	else descFloat = descriptors;

	VLADDescriptor queryDesc(descFloat, vDict);
	auto vlad = queryDesc.flattened();

	cv::Mat neighborsIdx;
	kdtree.findNearest(vlad, numToReturn, INT_MAX, neighborsIdx);

	return neighborsIdx;
}


bool
VLAD::save(const std::string &f)
{
	try {
		fstream indexFileFd;
		indexFileFd.open(f, fstream::out | fstream::trunc);
		if (!indexFileFd.is_open())
			throw runtime_error("Unable to create map file");

		boost::archive::binary_oarchive vladfd(indexFileFd);
		vladfd << vDict.centers;
		vladfd << vDescriptors;
		vladfd << kdtree;

		indexFileFd.close();
	} catch (exception &e) {
		return false;
	}

	return true;
}


bool
VLAD::load(const std::string &f)
{
	try {
		fstream indexFileFd;
		indexFileFd.open(f, fstream::in);
		if (!indexFileFd.is_open())
			throw runtime_error("Unable to create map file");

		boost::archive::binary_iarchive vladfd(indexFileFd);
		vladfd >> vDict.centers;
		vladfd >> vDescriptors;
		vladfd >> kdtree;

		indexFileFd.close();
	} catch (exception &e) {
		return false;
	}

	return true;
}


void
VLAD::initClusterCenters(const cv::Mat &cluster_centers)
{
	vDict.setCenters(cluster_centers);
}


void
VLAD::stopTrain()
{
	auto hasTrained = !(vDescriptors.size()==0);

	cout << "Cluster center adaptation\n";
	cv::Mat oldDict = vDict.centers.clone();
	auto hugeDescMat = createMatFromVectorMat(trainDescriptors);
	vDict.adapt(hugeDescMat);

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

	cv::Mat_<int> responses(1, D.rows);
	std::iota(responses.begin(), responses.end(), 0);

	kdtree.build(D);
}


} /* namespace PlaceRecognizer */
