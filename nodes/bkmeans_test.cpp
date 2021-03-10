#include <iostream>
#include <random>
#include <limits>
#include <numeric>
#include <vector>
#include <Eigen/Sparse>

#include "opencv2/core.hpp"
#include "VLAD.h"

using namespace std;
using namespace PlaceRecognizer;


void test_matrix_op()
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
}


typedef Eigen::Triplet<float> T;

void test_sparse_op()
{
	Eigen::SparseMatrix<float, Eigen::RowMajor> spm(4,4);

	vector<T> coeffs;
	coeffs.push_back({0, 0, 5});
	coeffs.push_back({1, 1, 8});
	coeffs.push_back({2, 2, 3});
	coeffs.push_back({3, 1, 6});

	spm.setFromTriplets(coeffs.begin(), coeffs.end());

	auto onx = spm.outerIndexPtr();
	for (int i=0; i<spm.outerSize(); ++i)
		cout << onx[i] << endl;

	cout << spm << endl;

	auto inx = spm.innerIndexPtr();
	for (int i=0; i<spm.innerSize(); ++i)
		cout << inx[i] << endl;
}


int main(int argc, char *argv[])
{
	test_sparse_op();
	return 0;
}
