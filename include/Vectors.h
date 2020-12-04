/*
 * Vectors.h
 *
 *  Created on: Nov 17, 2020
 *      Author: sujiwo
 */

#ifndef PLACE_RECOGNIZER_VECTORS_H_
#define PLACE_RECOGNIZER_VECTORS_H_

#include <nmmintrin.h>
#include <type_traits>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>


namespace PlaceRecognizer {

typedef cv::Mat_<float> Matf;
typedef cv::Mat_<double> Matd;
typedef cv::Mat_<cv::Vec3f> Matf3;
typedef cv::Mat_<int> Mati;
// Unsigned 32-bit Integer is not supported by OpenCV
typedef cv::Mat_<uint> Matui;
typedef cv::Mat_<bool> Matb;
typedef cv::Mat_<unsigned char> Matc;
typedef cv::Mat_<cv::Vec3b> Matc3;


inline const uint64_t &Take64(const cv::Mat &R, uint n)
{
	uint64_t *dt = (uint64_t*)R.data;
	return dt[n];
}

inline uint64_t &Take64(cv::Mat &R, uint n)
{
	uint64_t *dt = (uint64_t*)R.data;
	return dt[n];
}

uint inline hamming_distance(const cv::Mat_<uint8_t> &X, const cv::Mat_<uint8_t> &Y)
{
	uint d=0;
	assert(X.cols*X.rows==Y.cols*Y.rows);
	for (int i=0; i<(X.rows*X.cols)/8; i++) {
		uint64_t *x = (uint64_t*)(X.data + i*8),
				 *y = (uint64_t*)(Y.data + i*8);
		d += _mm_popcnt_u64(*x ^ *y);
	}
	return d;
}

uint inline hamming_distance(const cv::Mat &X, const cv::Mat &Y)
{
	uint d=0;
	assert(X.cols==Y.cols and X.cols%8==0);
	for (int i=0; i<X.cols/8; i++) {
		uint64_t *x = (uint64_t*)(X.data + i*8),
				 *y = (uint64_t*)(Y.data + i*8);
		d += _mm_popcnt_u64(*x ^ *y);
	}
	return d;
}

template<typename I>
uint HammingDistance(const cv::Mat_<I> &v1, const cv::Mat_<I> &v2)
{
	assert(std::is_integral<I>::value and (v1.rows*v1.cols==v2.rows*v2.cols));


}


}	// namespace PlaceRecognizer


#endif /* PLACE_RECOGNIZER_VECTORS_H_ */
