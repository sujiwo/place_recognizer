/*
 * recognizer_srv.cpp
 *
 *  Created on: Sep 29, 2020
 *      Author: sujiwo
 */


#include <iostream>
#include "ros/ros.h"
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "IncrementalBoW.h"
#include "ProgramOptionParser.h"
#include "place_recognizer/place_recognizer.h"

#ifdef SEGNET_FOUND
#include "Segmentation.h"
#endif


using namespace std;


class RecognizerService
{
public:
RecognizerService(ros::NodeHandle &node, PrgOpt::ProgramOption &opt)
{
	binFeats = cv::ORB::create(
			opt.get<int>("numfeats", 6000),
			1.2,
			8,
			31,
			0,
			2,
			cv::ORB::HARRIS_SCORE,
			31,
			10);
	imagedb.loadFromDisk(opt.get<string>("mapfile", ""));
	placeRecognSrv = node.advertiseService("place_recognizer", &RecognizerService::service, this);

#ifdef SEGNET_FOUND
	auto segnetModel = opt.get<string>("segnet-model", ""),
		segnetWeight = opt.get<string>("segnet-weight", "");
	if (segnetModel.empty()==false and segnetWeight.empty()==false) {
		gSegment.reset(new PlaceRecognizer::Segmentation(segnetModel, segnetWeight));
		cout << "SegNet is used" << endl;
	}
#endif

	cout << "Ready\n";
}

bool service(
	place_recognizer::place_recognizer::Request &request,
	place_recognizer::place_recognizer::Response &response)
{
	auto image = cv_bridge::toCvShare(request.input, nullptr, "bgr8")->image;
	std::vector<cv::KeyPoint> kpList;
	cv::Mat descriptors;
	binFeats->detectAndCompute(image, cv::Mat(), kpList, descriptors, false);

	vector<vector<cv::DMatch>> featureMatches;
	imagedb.searchDescriptors(descriptors, featureMatches, 2, 32);
	// Filter matches according to ratio test
	vector<cv::DMatch> realMatches;
	for (uint m=0; m<featureMatches.size(); m++) {
		if (featureMatches[m][0].distance < featureMatches[m][1].distance * 0.65)
			realMatches.push_back(featureMatches[m][0]);
	}

	vector<PlaceRecognizer::ImageMatch> imageMatches;
	imagedb.searchImages(descriptors, realMatches, imageMatches);
	response.keyframeId.clear();
	for (int i=0; i<min(15, (int)imageMatches.size()); i++) {
		response.keyframeId.push_back(imageMatches[i].image_id);
	}

	return true;
}

static
PrgOpt::ProgramOption
prepare_options()
{
	PrgOpt::ProgramOption opts;
	opts.addSimpleOptions("mapfile", "Map file input path");
	opts.addSimpleOptions<int>("numfeats", "Number of features from single image");

#ifdef SEGNET_FOUND
	opts.addSimpleOptions("segnet-model", "Path to SegNet model file");
	opts.addSimpleOptions("segnet-weight", "Path to SegNet weight file");
#endif

	return opts;
}

private:
	PlaceRecognizer::IncrementalBoW imagedb;
	ros::ServiceServer placeRecognSrv;
	cv::Ptr<cv::FeatureDetector> binFeats;

#ifdef SEGNET_FOUND
	std::shared_ptr<PlaceRecognizer::Segmentation> gSegment=nullptr;
#endif

};


int main(int argc, char *argv[])
{
	auto opts = RecognizerService::prepare_options();
	opts.parseCommandLineArgs(argc, argv);

	ros::init(argc, argv, "place_recognizer");
	ros::NodeHandle node;

	RecognizerService srv(node, opts);
	ros::spin();

	return 0;
}
