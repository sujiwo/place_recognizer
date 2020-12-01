#include <iostream>
#include "BKMeans.h"
#include "npy.hpp"
#include "opencv2/core.hpp"

using namespace std;
using namespace PlaceRecognizer;

int main(int argc, char *argv[])
{
	BKMeans bkm(256, 20);
	auto descriptors = npy::loadMat(argv[1]);
	cout << descriptors.rows << 'x' << descriptors.cols << endl;

	auto i = bkm.cluster(descriptors);
	cout << "Output: " << i << endl;
	auto C = bkm.get_centroids();
	npy::saveMat(C, "/tmp/BKMeans.out.npy");

	return 0;
}
