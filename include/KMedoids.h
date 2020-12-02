/*
 * KMedoids.h
 *
 *  Created on: Nov 11, 2020
 *      Author: sujiwo
 */

#ifndef PLACE_RECOGNIZER_KMEDOIDS_H_
#define PLACE_RECOGNIZER_KMEDOIDS_H_

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/hal.hpp>


namespace PlaceRecognizer
{

class KMedoids
{
public:
	KMedoids(uint numOfClusters, uint iteration);
	virtual ~KMedoids();

	bool cluster(cv::InputArray M);

	cv::Mat get_medoids() const
	{ return medoids.clone(); }

protected:
	typedef cv::Mat_<uint8_t> BinaryData;

	const uint
		numOfClusters,
		iteration;

	BinaryData samples;
	BinaryData medoids;
};

} /* namespace PlaceRecognizer */

#endif /* PLACE_RECOGNIZER_LIB_KMEDOIDS_H_ */
