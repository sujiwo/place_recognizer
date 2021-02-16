#!/usr/bin/python

from argparse import ArgumentParser, ArgumentError
from place_recognizer import srv
from place_recognizer import GenericImageDatabase
from place_recognizer.GenericImageMap import getEnhancementMethods
import rospy
import cv_bridge
import cv2
from geometry_msgs.msg import Point


class PlaceRecognizerServer():
    def __init__(self):
        _parser = ArgumentParser(description="ROS Place Recognition Service")
        _parser.add_argument("mapfile", type=str, help="Path to map file to be loaded")
        _parser.add_argument("--numret", type=int, default=5, help="Number of candidate positions to be returned")
        
        enhanceMethods = getEnhancementMethods()
        if len(enhanceMethods)!=0:
            _parser.add_argument("--ime", type=str, choices=enhanceMethods, help='Preprocess image with this method')
        else:
            print("Enhancement not available; install im_enhance if you want")
    
        prog_arguments = _parser.parse_args()
        
        self.bridge = cv_bridge.CvBridge()
        self.imagedb = GenericImageDatabase(prog_arguments.mapfile)
        self.imagedb.show_image_frame = False
        self.numReturn = prog_arguments.numret
        
        rospy.init_node('place_recognizer_server')
        self.process = rospy.Service('place_recognizer', srv.place_recognizer, self.serve)
        print("Ready")
        rospy.spin()
    
    def serve(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg.input, desired_encoding='bgr8')
        
        q_ans = self.imagedb.query(image, self.numReturn)
        pointList = [Point(qa[0], qa[1], qa[2]) for qa in q_ans]
        
        return srv.place_recognizerResponse(pointList)


def server_main():
    _parser = ArgumentParser(description="ROS Place Recognition Service")
    _parser.add_argument("mapfile", type=str)
    
    enhanceMethods = getEnhancementMethods()
    if len(enhanceMethods)!=0:
        _parser.add_argument("--ime", type=str, choices=enhanceMethods, help='Preprocess image with this method')
    else:
        print("Enhancement not available; install im_enhance if you want")

    prog_arguments = _parser.parse_args()
    rospy.init_node('place_recognizer_server')
    server = rospy.Service('place_recognizer', srv.place_recognizer, server_run_place_recognizer)
    print("Ready")
    rospy.spin()


if __name__=='__main__' :
    server = PlaceRecognizerServer()
    
