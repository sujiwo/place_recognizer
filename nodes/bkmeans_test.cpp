#include <iostream>
#include <random>
#include <limits>
#include <numeric>
#include <vector>

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

	cv::Mat R;
	cv::repeat(I, 5, 1, R);

	cv::Mat_<int> Seq(1, 5);
	std::iota(Seq.begin(), Seq.end(), 0);

	cout << M << endl;
	cout << endl;
	cout << R << endl;
	cout << endl;
	cout << Seq << endl;
	return 0;
}
