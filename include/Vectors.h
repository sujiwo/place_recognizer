/*
 * Vectors.h
 *
 *  Created on: Nov 17, 2020
 *      Author: sujiwo
 */

#ifndef PLACE_RECOGNIZER_VECTORS_H_
#define PLACE_RECOGNIZER_VECTORS_H_

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

template<typename I>
uint HammingDistance(const cv::Mat_<I> &v1, const cv::Mat_<I> &v2)
{
	assert(std::is_integral<I>::value and (v1.rows*v1.cols==v2.rows*v2.cols));


}


}	// namespace PlaceRecognizer


#endif /* PLACE_RECOGNIZER_VECTORS_H_ */
