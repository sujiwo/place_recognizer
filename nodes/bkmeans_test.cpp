#include <iostream>
#include "pam.h"
#include "npy.hpp"
#include <random>
#include <limits>

#include "opencv2/core.hpp"

using namespace std;
using namespace PlaceRecognizer;

int main(int argc, char *argv[])
{
	auto descriptors = npy::loadMat(argv[1]);
	int K = atoi(argv[2]);
	cout << descriptors.rows << 'x' << descriptors.cols << endl;

	CvDistMatrix CM(descriptors);
	LAB pamInit(&CM);

	FastPAM kmediods(descriptors.rows, &CM, &pamInit, K, 300, 0);
	auto d = kmediods.run();

	auto md = kmediods.getMedoids();
	cv::Mat medoids(md.size(), descriptors.cols, CV_8UC1);
	for (int i=0; i<md.size(); i++) {
		descriptors.row(md[i]).copyTo(medoids.row(i));
	}
	npy::saveMat(medoids, "/tmp/medoids.npy");

	return 0;
}
