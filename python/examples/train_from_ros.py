#!/usr/bin/env python2
"""
This script trains a place_recognizer map file from ROS image topic
"""
import rospy
import time
import sys
import cv2
from argparse import ArgumentParser, ArgumentError
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped, PointStamped
from place_recognizer import VLAD2, VisualDictionary, IncrementalBoW


class ImageSubscriber(rospy.Subscriber):
    """
    Emulates image_transport for Python
    XXX: This class may have to be moved to a distinct module
    """
    def __init__(self, imageTopic, _callback):
        pass
    
    def callback(self, msg):
        pass


class RosTrainer:
    def __init__(self, image_topic, position_topic, method, mapfile_output, mapfile_load=None, vdictionaryPath=None):
        self.method = method
        self.mapfile_output = mapfile_output
        
        # Prepare the map
        if mapfile_load is not None:
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
        self.imgSubscriber = rospy.Subscriber(image_topic, Image, self.imageCallback)
        
        
    def imageCallback(self, msg):
        pass
    
    def positionCallback(self, posMsg):
        pass

def main():
    
    parser = ArgumentParser(description="Place Recognizer Training from ROS messages")
    parser.add_argument("topic", type=str, help="Image topic to subscribe to")
    parser.add_argument("output", type=str, help="Map file to save to")
    parser.add_argument("--dictionary", type=str, metavar="path", help="Path to initial visual dictionary (only for vlad)")
    parser.add_argument("--method", type=str, choices=['vlad', 'ibow'], default='vlad', help="Choices of method for training")
    parser.add_argument("--resize", type=float, metavar="ratio", default=0.53333, help="Rescale image size with this ratio")
    parser.add_argument("--load", type=str, metavar="path", default="", help="Load previous map for retrain")
    parser.add_argument("--position", type=str, metavar="pos_topic", help="Topic for position sources")
    
    parser.parse_args()
    
    while True:
        try:
            time.sleep(0.5)
            print("Press CTRL+C to quit")
        except KeyboardInterrupt:
            break
    
    print("Exiting")
    exit(0)    

if __name__=="__main__":
    main()
