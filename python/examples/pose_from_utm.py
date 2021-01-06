#!/usr/bin/env python2
# This is a simple ROS node for localization using GNSS messages
# Two types of message are supported:
# 


from geodesy import utm
import numpy as np
import rospy
import pynmea2
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import NavSatFix
from nmea_msgs.msg import Sentence
from argparse import ArgumentParser
from importlib import import_module


publisher = None
publishTopicName = '/gnss_pose'


class MsgListener:
    def __init__ (self, topic):
        self.initsub = rospy.Subscriber(topic, rospy.AnyMsg, self.initial_callback)
        self.topic = topic
    
    def initial_callback(self, msg0):
        connection_header = msg0._connection_header['type'].split('/')
        ros_pkg = connection_header[0] + '.msg'
        msg_type = connection_header[1]
        msg_class = getattr(import_module(ros_pkg), msg_type)
        self.initsub.unregister()
        self.deserialized_sub = rospy.Subscriber(self.topic, msg_class, self.nextMessageCallback)
        # Only run once
        msg0f = msg_class()
        self.nextMessageCallback(msg0f.deserialize(msg0._buff))
    
    def nextMessageCallback(self, msg):
        pass


class LocalizerGnss(MsgListener):
    def nextMessageCallback(self, msg):
        if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
            xycoord = utm.fromLatLong(msg.latitude, msg.longitude, msg.altitude)
        elif hasattr(msg, 'sentence'):
            try:
                mgeo = pynmea2.parse(msg.sentence)
                xycoord = utm.fromLatLong(float(mgeo.latitude), float(mgeo.longitude), float(mgeo.altitude))
            except:
                return
        pxy = PointStamped()
        pxy.header = msg.header
        pxy.point.x = xycoord.easting
        pxy.point.y = xycoord.northing
        pxy.point.z = xycoord.altitude
        publisher.publish(pxy)
        



if __name__=="__main__":
    
    parser = ArgumentParser(description="Simple GNSS Localizer")
    parser.add_argument('topic', type=str, help="Source topic for conversion from Lat/Lon to X/Y")
    parser.add_argument('--dst', type=str, default=publishTopicName, help='Destination topic to publish to')
    cmdArgs = parser.parse_args()
    
    rospy.init_node('geocoordinate', anonymous=True)
    publisher = rospy.Publisher(cmdArgs.dst, PointStamped, queue_size=100)
    handler = LocalizerGnss(cmdArgs.topic)
    rospy.spin()
    
    pass
