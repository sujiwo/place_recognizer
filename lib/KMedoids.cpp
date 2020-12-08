/*
 * KMedoids.cpp
 *
 *  Created on: Nov 11, 2020
 *      Author: sujiwo
 */

#include <random>
#include <vector>
#include <set>
#include <algorithm>
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


void
KMedoids::getClusterMedoids()
{

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

	initialize(clusterIds);


	return true;
}


/*
 * Implementation of LAB (Linear Approximative Build)
 */
void
KMedoids::initialize(std::vector<uint> &ids)
{
	// Initialize RNG first

    int nn = (int) ids.size();
    std::vector<uint> medids;
    std::set<uint> medids_set;

    // O(sqrt(n)) sample if k^2 < n.
    int ssize = 10 + (int)ceil(std::sqrt((double)nn));
    if (ssize > nn) ssize = nn;

    // We need three temporary storage arrays:
    vector<double>
    	mindist(nn, DBL_MIN),
    	bestd(nn),
		tempd(nn, DBL_MIN),
		tmp;
    vector<uint> sample(nn);
    std::copy(ids.begin(), ids.end(), sample.begin());
    int range = (int)sample.size();
    std::random_shuffle(sample.begin(), sample.end());
}


} /* namespace PlaceRecognizer */
