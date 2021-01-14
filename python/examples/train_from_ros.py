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
import numpy as np
from copy import copy
from argparse import ArgumentParser, ArgumentError
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped, PointStamped
from place_recognizer import VLAD2, VisualDictionary, IncrementalBoW, ImageSubscriber, GenericTrainer


class RosTrainer(GenericTrainer):
    def __init__(self, image_topic, position_topic, method, mapfile_output, mapfile_load=None, vdictionaryPath=None):
        super(RosTrainer, self).__init__(method, mapfile_output, mapfile_load, vdictionaryPath)

        # Prepare ROS subsystem
        rospy.init_node("place_recognizer_trainer", disable_signals=True)
        self.imgSubscriber = ImageSubscriber(image_topic, self.imageCallback)
        
        self.currentPosition = None
        if (position_topic!=''):
            self.positionSub = rospy.Subscriber(position_topic, PointStamped, self.positionCallback)
            
        self.initTrain()

    
    def imageCallback(self, image):
        self.addImage(image, copy(self.currentPosition))
    
    def positionCallback(self, pos_message):
        self.currentPosition = np.array([
            pos_message.point.x, 
            pos_message.point.y, 
            pos_message.point.z ]) 

    def stopTrain(self):
        # XXX: Do something with missing positions
        
        super(RosTrainer, self).stopTrain()

        

def main():
    
    parser = ArgumentParser(description="Place Recognizer Training from ROS messages")
    parser.add_argument("topic", type=str, help="Image topic to subscribe to")
    parser.add_argument("output", type=str, help="Map file to save to")
    parser.add_argument("--dictionary", type=str, metavar="path", help="Path to initial visual dictionary (only for vlad)")
    parser.add_argument("--method", type=str, choices=['vlad', 'ibow'], default='vlad', help="Choices of method for training")
    parser.add_argument("--resize", type=float, metavar="ratio", default=GenericTrainer.resize_factor, help="Rescale image size with this ratio")
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
        
    trainer.stopTrain()    
    print("Exiting")
    exit(0)    

if __name__=="__main__":
    main()
