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
#include <opencv2/ml.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include "IncrementalBoW.h"


namespace PlaceRecognizer {


struct VisualDictionary
{
	friend class VLAD;
	friend struct VLADDescriptor;

	VisualDictionary(uint _numWords=256) :
		numWords(_numWords)
	{}

	void setCenters(const cv::Mat& precomputedCenters);

	VisualDictionary(const cv::Mat& precomputedCenters)
	{ setCenters(precomputedCenters); }

	bool build (cv::Mat &descriptors);

	/*
	 * Returns a list of centroids for a set of descriptors
	 */
	std::vector<uint>
	predict(const cv::Mat &imageDescriptors) const;

	cv::Mat
	adapt(cv::InputArray newDescriptors, bool dryRun=false);

	/*
	 * Returns descriptors for the centroids
	 */
	inline cv::Mat getCenters() const
	{ return centers; }

protected:
	uint numWords;
	cv::Mat centers;

	uint predict1row(const cv::Mat &descriptors, int rowNum) const;
};


struct VLADDescriptor
{
	VLADDescriptor(const cv::InputArray imageDescriptors, const VisualDictionary &dict)
	{
		auto M = imageDescriptors.getMat();
		compute(M, dict);
	}

	void compute(const cv::Mat &imageDescriptors, const VisualDictionary &dict);

	cv::Mat normalized() const;

	cv::Mat flattened() const;

	void adaptNewCentroids(const VisualDictionary &newDict, const cv::Mat &oldCentroids);

	// The resulting aggregated (VLAD) descriptor, unnormalized
	cv::Mat descriptors;
	std::vector<uint> centroid_counters;
};



class VLAD {
public:
	VLAD(uint numWords=256, uint leafSize=40);
	virtual ~VLAD();

	void initTrain();

	void initClusterCenters(const cv::Mat &cluster_centers);

	void addImage(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints, uint imageId=-1);

	void stopTrain();

	std::vector<uint> query(const cv::Mat &descriptors, const uint numToReturn) const;

	bool save(const std::string &path);
	bool load(const std::string &path);

	uint lastImageId() const
	{ return vDescriptors.size(); }

protected:
	uint leafSize;
	VisualDictionary vDict;
	std::vector<uint> imageIds;
	cv::Ptr<cv::ml::KNearest> searchTree;

	// training data structures
	std::vector<cv::Mat> trainDescriptors;
	std::vector<std::pair<uint,uint>> trainDescriptorPtr;

	std::vector<VLADDescriptor> vDescriptors;

	cv::Mat computeVlad(const cv::Mat &descriptors) const;

	void rebuildVladDescriptors(const cv::Mat &oldDict);

	cv::Mat flatNormalDescriptors() const;
};

} /* namespace PlaceRecognizer */

#endif /* PLACE_RECOGNIZER_VLAD_H_ */
