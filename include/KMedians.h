/*
 * KMedians.h
 *
 *  Created on: Dec 2, 2020
 *      Author: sujiwo
 */

#ifndef PLACE_RECOGNIZER_KMEDIANS_H_
#define PLACE_RECOGNIZER_KMEDIANS_H_

#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>

namespace PlaceRecognizer {

class KMedians {
public:

	KMedians(int numOfClusters, int iteration);

	virtual ~KMedians()
	{}

	bool cluster(cv::Mat &data);

	cv::Mat get_centroids() const
	{ return centroids.clone(); }

protected:
	const int numOfClusters, iteration;
	cv::Mat samples;

	cv::Mat centroids;

	const uint bit_width;
};

} /* namespace PlaceRecognizer */

#endif /* PLACE_RECOGNIZER_KMEDIANS_H_ */
