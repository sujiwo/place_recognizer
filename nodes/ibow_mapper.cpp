/*
 * Mapper node for creating visual maps of places
 */

#include <string>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/features2d.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "IncrementalBoW.h"
#include "ProgramOptionParser.h"
#include "ImageBag.h"

#ifdef SEGNET_FOUND
#include "Segmentation.h"
#endif

#include "npy.hpp"


using namespace std;
namespace po=boost::program_options;
using Path = boost::filesystem::path;


class IBoW_Mapper_App
{
public:

static const int defaultNumOfFeatures = 3000;
static constexpr float defaultSampleImageRate = 7.5;

IBoW_Mapper_App(int argc, char *argv[])
{
	auto options = prepare_options();
	options.parseCommandLineArgs(argc, argv);

	auto numFeats = options.get<int>("numfeats", defaultNumOfFeatures);
	featureDetector = cv::ORB::create(numFeats);

	bagFd.open(options.get<string>("bagfile", ""), rosbag::BagMode::Read);

	auto imageTopic = options.get<string>("topic", "");
	if (imageTopic.empty())
		imageTopic = PlaceRecognizer::ImageBag::suggestTopic(bagFd);
	imageBag.reset(new PlaceRecognizer::ImageBag(bagFd, imageTopic));
	cout << "Using `" << imageBag->getTopic() << "' as image topic" << endl;

	startTimeSeconds = options.get<float>("start-time", startTimeSeconds);
	maxSecondsFromStart = options.get<float>("stop-time", maxSecondsFromStart);
	imageBag->setTimeConstraint(startTimeSeconds, maxSecondsFromStart);
	imageBag->desample(options.get<float>("desample", defaultSampleImageRate), messageList);
	cout << "# of target frames: " << imageBag->size() << endl;

	// Dump message IDs by using Numpy
	npy::saveMat(messageList, "/tmp/debugmapper.log");

	auto _mapOutputPath = options.get<string>("mapfile", "");
	if (_mapOutputPath.empty()) {
		//
	}
	else mapOutputPath = Path(_mapOutputPath);

	ros::init(argc, argv, "bow_mapper");
	ros::Time::init();
	rosEnabled = ros::master::check();
	if (rosEnabled) {
		cout << "ROS connector enabled" << endl;
		hdl.reset(new ros::NodeHandle);
		imageTransport.reset(new image_transport::ImageTransport(*hdl));
		keyptPublisher = imageTransport->advertise("bow_mapper_image", 10);
	}

#ifdef SEGNET_FOUND
	auto segnetModel = options.get<string>("segnet-model", ""),
		segnetWeight = options.get<string>("segnet-weight", "");
	if (segnetModel.empty()==false and segnetWeight.empty()==false) {
		gSegment.reset(new PlaceRecognizer::Segmentation(segnetModel, segnetWeight));
		cout << "SegNet is used" << endl;
	}
#endif

	return;
}


~IBoW_Mapper_App()
{
	imageTransport.reset();
	hdl->shutdown();
}


void publishToRos(const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints)
{
	if (rosEnabled==false)
		return;

	cv::Mat drawKpts;
	cv::drawKeypoints(image, keypoints, drawKpts, cv::Scalar(0,255,0));

	cv_bridge::CvImage cvImg;
	cvImg.encoding = sensor_msgs::image_encodings::BGR8;
	cvImg.image = drawKpts;
	cvImg.header.stamp = ros::Time::now();
	keyptPublisher.publish(*cvImg.toImageMsg());
}


void run()
{
	for (auto mId: messageList) {
		auto frameImg = imageBag->at(mId);

		cv::Mat mask;
#ifdef SEGNET_FOUND
		if (gSegment!=nullptr) {
			mask = gSegment->buildMask(frameImg);
			cv::resize(mask, mask, frameImg.size(), 0, 0, cv::INTER_NEAREST);
		}
#endif

		std::vector<cv::KeyPoint> kpList;
		cv::Mat descriptors;
		featureDetector->detectAndCompute(frameImg, mask, kpList, descriptors, false);

		if (mId==messageList.front()) {
			mapperProc.addImage(mId, kpList, descriptors);
		}
		else {
			mapperProc.addImage2(mId, kpList, descriptors);
		}

		publishToRos(frameImg, kpList);
		cout << mId << "/" << imageBag->size() << "\r" << flush;
	}
	cout << endl;

	cout << "Saving to " << mapOutputPath.string() << "... ";
	mapperProc.saveToDisk(mapOutputPath.string());
	cout << "Done\n";

//	mapperProc.
}

static
PrgOpt::ProgramOption
prepare_options()
{
	PrgOpt::ProgramOption opts;

	// XXX: put bagfile as positional argument
	opts.addSimpleOptions("bagfile", "Path to bagfile to be read");
	opts.addSimpleOptions<int>
		("numfeats", string("Number of features from single image; default is "+to_string(defaultNumOfFeatures)));
	opts.addSimpleOptions<float>
		("desample", "Reduce sample frequency of the bag; default is "+to_string(defaultSampleImageRate));
	opts.addSimpleOptions("topic", "Image topic from bag");
	opts.addSimpleOptions
		<decltype(IBoW_Mapper_App::startTimeSeconds)>
		("start-time", "Seconds from start of bag time");
	opts.addSimpleOptions
		<decltype(IBoW_Mapper_App::startTimeSeconds)>
		("stop-time", "Maximum seconds from start");
	opts.addSimpleOptions("mapfile", "Map file output path");

#ifdef SEGNET_FOUND
	opts.addSimpleOptions("segnet-model", "Path to SegNet model file");
	opts.addSimpleOptions("segnet-weight", "Path to SegNet weight file");
#endif
	return opts;
}

private:
	cv::Ptr<cv::Feature2D> featureDetector;
	rosbag::Bag bagFd;
	PlaceRecognizer::ImageBag::Ptr imageBag;
	string outputMapFilename;
	RandomAccessBag::DesampledMessageList messageList;
	bool isCompressedImage=false;

#ifdef SEGNET_FOUND
	std::shared_ptr<PlaceRecognizer::Segmentation> gSegment=nullptr;
#endif

	PlaceRecognizer::IncrementalBoW mapperProc;

	// Time constraint for bag
	float startTimeSeconds=0,
		maxSecondsFromStart=-1;

	// ROS Part
	bool rosEnabled = false;
	std::shared_ptr<ros::NodeHandle> hdl=nullptr;
	std::shared_ptr<image_transport::ImageTransport> imageTransport=nullptr;
	image_transport::Publisher keyptPublisher;

	Path mapOutputPath;
};


int main(int argc, char *argv[])
{
	IBoW_Mapper_App mapper(argc, argv);
	mapper.run();

	return 0;
}
