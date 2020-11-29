/*
 * VLAD.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: sujiwo
 */

#include "VLAD.h"
#include "BOWKmajorityTrainer.h"


namespace PlaceRecognizer {

bool
VisualDictionary::build (cv::Mat &descriptors)
{
	cv::BOWKmajorityTrainer trainer(numWords);
	centers = trainer.cluster(descriptors);
	return (centers.rows==numWords);
}


bool
VisualDictionary::rebuild (cv::Mat &descriptors)
{

}


cv::Mat
VisualDictionary::predict (const cv::Mat &imageDescriptors)
{

}


VLAD::VLAD() {
	// TODO Auto-generated constructor stub

}

VLAD::~VLAD() {
	// TODO Auto-generated destructor stub
}

} /* namespace PlaceRecognizer */
