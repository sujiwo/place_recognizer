/*
 * KMedoids.cpp
 *
 *  Created on: Nov 11, 2020
 *      Author: sujiwo
 */

#include "KMedoids.h"
#include "Vectors.h"


namespace PlaceRecognizer {

KMedoids::KMedoids(uint N, uint I) :
	numOfClusters(N),
	iteration(I)
{}


KMedoids::~KMedoids() {}


bool
KMedoids::cluster(cv::InputArray M)
{
	auto binary_data = M.getMat();

	// Check & preparation
	// Make sure data type is unsigned 8-bit integer
	// with bit-width as multiples of 64
	int m_type = binary_data.type() & CV_MAT_DEPTH_MASK;
	assert(m_type==cv::DataType<uint8_t>::type);
	assert(binary_data.channels()==1);

	samples = binary_data;

	return true;
}

} /* namespace PlaceRecognizer */
