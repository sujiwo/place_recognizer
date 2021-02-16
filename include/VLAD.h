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
#include "kdtree.hpp"


namespace PlaceRecognizer {


struct VisualDictionary
/*
 * Descriptor compression to clusters
 * Note: also available in Python
 */
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
	 * Returns a list of centroids for a set of descriptors.
	 *
	 * @param imageDescriptors: features acquired from detector
	 * Should be in float32
	 */
	std::vector<uint>
	predict(const cv::Mat &imageDescriptors) const;

	/*
	 * Adjust cluster centers to new set of image descriptors
	 */
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
	typedef std::vector<int> Counters;

	VLADDescriptor()
	{}

	VLADDescriptor(const cv::Mat &imageDescriptors, const VisualDictionary &dict);

	VLADDescriptor(const std::vector<cv::Mat> imageDescriptors, const VisualDictionary &dict);

	void compute(const cv::Mat &imageDescriptors, const VisualDictionary &dict);

	cv::Mat normalized() const;

	cv::Mat flattened() const;

	void adaptNewCentroids(const VisualDictionary &newDict, const cv::Mat &oldCentroids);

	// The resulting aggregated (VLAD) descriptor, unnormalized
	cv::Mat descriptors;
	Counters centroid_counters;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int)
	{
		ar & descriptors;
		ar & centroid_counters;
	}
};



class VLAD {
public:
	VLAD(uint numWords=256, uint leafSize=40);
	virtual ~VLAD();

	void initTrain();

	void initClusterCenters(const cv::Mat &cluster_centers);

	void addImage(const cv::Mat &descriptors, const std::vector<cv::KeyPoint> &keypoints, uint imageId=-1);

	void stopTrain();

	/*
	 * Search images in database using descriptors
	 */
	std::vector<int> query(const cv::Mat &descriptors, const uint numToReturn) const;

	/*
	 * Save/load routine to disk
	 */
	bool save(const std::string &path);
	bool load(const std::string &path);

	uint lastImageId() const
	{ return vDescriptors.size(); }

protected:
	uint leafSize;
	VisualDictionary vDict;
	std::vector<uint> imageIds;
	cv::ml::KDTree kdtree;

	// training data structures. will be discarded after training
	std::vector<cv::Mat> trainDescriptors;
	std::vector<std::pair<uint,uint>> trainDescriptorPtr;

	std::vector<VLADDescriptor> vDescriptors;

	void rebuildVladDescriptors(const cv::Mat &oldDict);

	cv::Mat flatNormalDescriptors() const;
};

} /* namespace PlaceRecognizer */

#endif /* PLACE_RECOGNIZER_VLAD_H_ */
