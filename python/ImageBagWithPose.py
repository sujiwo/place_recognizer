#!/usr/bin/env python2

import rosbag
from RandomAccessBag import RandomAccessBag, ImageBag
import cv2
import sys
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentError
from copy import copy
from .GeographicCoordinate import GeographicTrajectory


class ImageBagWithPose(ImageBag):
    """
    Image source that returns image and pose at each index
    """
    def __init__ (self, bagfilePath, imageTopic=None, poseTopic=None, frequency=-1, startTime=0, stopTime=-1):
        if (isinstance(bagfilePath, str)):
            self.bag = rosbag.Bag(bagFd, mode="r")
        else:
            assert(type(bagfilePath)==rosbag.bag.Bag)
            self.bag = bagfilePath
        imageTopic = ImageBagWithPose.probeImageTopic(self.bag, imageTopic)[0]
        poseTopic = ImageBagWithPose.probeTrajectoryTopic(self.bag, poseTopic)[0]
        
        ImageBag.__init__(self, self.bag, imageTopic)
        self.desample(frequency, startTime=startTime, stopTime=stopTime)
        
        self.trajectoryBag = RandomAccessBag(self.bag, poseTopic, start_time, stopTime)
#         self.trajectory = GeographicTrajectory
    
    def __getitem__ (self, i):
        pass
    
    def close(self):
        self.bag.close()
        
    @staticmethod
    def probeImageTopic(bagFd, imageTopic=None):
        """
        Find any supported image topics from a bag file
        
        Parameters
        ----------
        bagFd: rosbag.Bag or string of path to bag file
        imageTopic: requested image topic to be checked
        
        Return
        ------
        List of any image topics in bag file. If imageTopic is not None, returns list with single member
        If imageTopic turns out to be not of image type, raise ValueError
        """
        allBagConns = RandomAccessBag.getAllConnections(bagFd)
        imgTopics = []
        for bg in allBagConns:
            if bg.type()=="sensor_msgs/Image" or bg.type()=="sensor_msgs/CompressedImage":
                imgTopics.append(bg.topic())
            elif bg.topic()==imageTopic:
                raise ValueError("Requested topic is not an image")
        if imageTopic in imgTopics:
            return [imageTopic]
        elif imageTopic is not None:
            raise ValueError("Requested image topic does not exist")
        return imgTopics
    
    @staticmethod
    def probeTrajectoryTopic(bagFd, trackTopic=None):
        """
        Find any supported trajectory topics from a bag file
        
        Parameters
        ----------
        bagFd: rosbag.Bag or string of path to bag file
        trackTopic: requested image topic to be checked
        
        Return
        ------
        List of any trajectory topics in bag file. If trackTopic is not None, returns list with single member
        If trackTopic turns out to be not supported, raise ValueError
        """
        allBagConns = RandomAccessBag.getAllConnections(bagFd)
        trajectoryTopics = []
        for bg in allBagConns:
            if bg.type() in GeographicTrajectory.supportedMsgTypes:
                trajectoryTopics.append(bg.topic())
            elif bg.topic()==trackTopic:
                raise ValueError("Requested topic is not supported")
        if trackTopic in trajectoryTopics:
            return [trackTopic]
        elif trackTopic is not None:
            raise ValueError("Requested trajectory topic does not exist")
        return trajectoryTopics
        
        return trajectoryTopics
    
    @staticmethod
    def probeBagForImageAndTrajectory(bagFilePath, imageTopic=None, trajectoryTopic=None):
        allBagConns = RandomAccessBag.getAllConnections(bagFilePath)
        trajectorySrc = None
        imageBag = None
        for bg in allBagConns:
            if bg.type()=="sensor_msgs/Image" or bg.type()=="sensor_msgs/CompressedImage":
                if imageTopic is None:
                    imageBag = ImageBag(bagFilePath, bg.topic())
                elif bg.topic()==imageTopic:
                    imageBag = ImageBag(bagfilePath, imageTopic)
            elif bg.type() in GeographicTrajectory.supportedMsgTypes:
                trajectorySrc = bg
        if (not imageBag):
            raise ArgumentError("Image topic is invalid")
        return imageBag, trajectorySrc
