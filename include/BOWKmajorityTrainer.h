/*
 * BOWKmajorityTrainer.h
 *
 *  Created on: Sep 26, 2013
 *      Author: andresf
 */

#ifndef BOWKMAJORITYTRAINER_H_
#define BOWKMAJORITYTRAINER_H_

#include <opencv2/features2d.hpp>
#include <opencv2/flann/linear_index.h>
#include <opencv2/flann/random.h>
#include <opencv2/flann/dist.h>


namespace cv {

typedef cvflann::Hamming<uchar> HammingDistance;
typedef cvflann::LinearIndex<HammingDistance> HammingIndex;


struct KMajority {
	static cv::Mat initCentroids(const cv::Mat &trainData, int numClusters);
	static void computeCentroids(const Mat &features, Mat &centroids,
		std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo);
	static void cumBitSum(const cv::Mat& feature, cv::Mat& accVector);
	static void majorityVoting(const cv::Mat& accVector, cv::Mat& result, const int& threshold);
	static bool quantize(cv::Ptr<HammingIndex> index, const Mat &descriptors,
	        std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo, const int numClusters);
	static void handleEmptyClusters(std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo, int numClusters, int numDatapoints);
};


class BOWKmajorityTrainer: public BOWTrainer {

public:
	BOWKmajorityTrainer(int clusterCount, int maxIterations=100);
	virtual ~BOWKmajorityTrainer();
	virtual Mat cluster() const;
	virtual Mat cluster(const Mat& descriptors) const;

protected:
	int numClusters;
	int maxIterations;
};

} /* namespace cv */
#endif /* BOWKMAJORITYTRAINER_H_ */
