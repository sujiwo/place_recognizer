/*
 * KMedoids.cpp
 *
 *  Created on: Nov 11, 2020
 *      Author: sujiwo
 */

#include <vector>
#include <exception>
#include "KMedoids.h"
#include "Vectors.h"

using namespace std;


namespace PlaceRecognizer {

KMedoids::KMedoids(uint N, uint I) :
	numOfClusters(N),
	iteration(I)
{}


KMedoids::~KMedoids() {}


bool
KMedoids::check_compatible(const cv::Mat &Inp)
{

}


bool
KMedoids::set_centroids(cv::InputArray M)
{

	medoids = M.getMat().clone();
}


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
	if (samples.rows < numOfClusters)
		throw runtime_error("Insufficient number of clusters");

	medoids = BinaryData::zeros(numOfClusters, samples.cols);
	vector<double> errors(numOfClusters, 0);
	vector<uint>
		clusterIds(samples.rows, -1),
		saved(samples.rows, -1);

	double error = std::numeric_limits<double>::max();
	int ipass = 0;

	do {	// start the loop
		double total = std::numeric_limits<double>::max();
		int counter = 0;
		int period = 10;

		if (counter % period == 0) {
			// save cluster assignment periodically
			std::copy(clusterIds.begin(), clusterIds.end(), saved.begin());
		}

	} while (++ipass < iteration);

	return true;
}

} /* namespace PlaceRecognizer */
