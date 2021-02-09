#include <iostream>
#include <random>
#include <limits>

#include "opencv2/core.hpp"
#include "VLAD.h"

using namespace std;
using namespace PlaceRecognizer;

int main(int argc, char *argv[])
{
	cv::Mat M = cv::Mat::zeros(5, 3, CV_32SC1),
		I = cv::Mat::ones(1, 3, CV_32SC1);

	// This statement has no effect
	M.row(0) = I;

	// This statement succeed
	M.row(0) = cv::Mat::ones(1, 3, CV_32SC1);

	// This also good
	I.copyTo(M.row(3));

	M = M.reshape(0, 1);

	cout << M << endl;
	return 0;
}
