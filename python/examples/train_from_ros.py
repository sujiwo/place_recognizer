#!/usr/bin/env python2
"""
This script trains a place_recognizer map file from ROS image topic

Notes on deployment:
Both mapping methods may eat significant amount of time per frame;
it is recommended to use message throttling tools (eg. topic_tools)
to reduce number of message coming into mapping node, and/or playing
the bag with reduced frequency.
"""
import rospy
import time
import sys
import cv2
from argparse import ArgumentParser, ArgumentError
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped, PointStamped
from place_recognizer import VLAD2, VisualDictionary, IncrementalBoW, ImageSubscriber





class RosTrainer:
    
    def __init__(self, image_topic, position_topic, method, mapfile_output, mapfile_load=None, vdictionaryPath=None):
        self.method = method
        self.mapfile_output = mapfile_output
        
        # Prepare the map
        if mapfile_load!='':
            if method=="vlad":
                self.mapper = VLAD2.load(mapfile_load)
                print("VLAD file loaded")
            elif method=="ibow":
                self.mapper = IncrementalBoW()
                self.mapper.load(mapfile_load)
        else:
            if method=="vlad":
                vdict = VisualDictionary.load(vdictionaryPath)
                self.mapper = VLAD2(vdict)
            elif method=="ibow":
                self.mapper = IncrementalBoW()
        self.extractor = cv2.ORB_create(6000)
                
        # Prepare ROS subsystem
        rospy.init_node("place_recognizer_trainer", disable_signals=True)
        self.imgSubscriber = ImageSubscriber(image_topic, self.imageCallback)
        
        self.currentPosition = None
        if (position_topic!=''):
            self.positionSub = rospy.Subscriber(position_topic, PointStamped, self.positionCallback)
        
    def preprocess(self, image_message):
        return image_message
        
    def imageCallback(self, image_message):
        image_prep = self.preprocess(image_message)
        keypoints, descriptors = self.extractor.detectAndCompute(image_message, None)
#         self.mapper.addImage(imageId, descriptors, keypoints)
        cv2.imshow('image', image_message)
        cv2.waitKey(1)

        pass
    
    def positionCallback(self, posMsg):
        pass
    
    def stopTraining(self):
        self.imgSubscriber.unregister()
        print("Saving...")
#         self.mapper.save(self.mapfile_output)
        

def main():
    
    parser = ArgumentParser(description="Place Recognizer Training from ROS messages")
    parser.add_argument("topic", type=str, help="Image topic to subscribe to")
    parser.add_argument("output", type=str, help="Map file to save to")
    parser.add_argument("--dictionary", type=str, metavar="path", help="Path to initial visual dictionary (only for vlad)")
    parser.add_argument("--method", type=str, choices=['vlad', 'ibow'], default='vlad', help="Choices of method for training")
    parser.add_argument("--resize", type=float, metavar="ratio", default=0.53333, help="Rescale image size with this ratio")
    parser.add_argument("--load", type=str, metavar="path", default="", help="Load previous map for retrain")
    parser.add_argument("--position", type=str, metavar="pos_topic", default='', help="Topic for position sources")
    
    cmdArgs = parser.parse_args()
    
    trainer = RosTrainer(cmdArgs.topic, cmdArgs.position, cmdArgs.method, cmdArgs.output, cmdArgs.load, cmdArgs.dictionary)
    
    try:
        print("Press CTRL+C to quit")
        rospy.spin()
    except KeyboardInterrupt:
        print("Break pressed")
        print("Done")
        
    trainer.stopTraining()    
    print("Exiting")
    exit(0)    

if __name__=="__main__":
    main()
