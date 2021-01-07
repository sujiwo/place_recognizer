#!/usr/bin/env python2
import rospy
import cv_bridge
from importlib import import_module
from sensor_msgs.msg import Image, CompressedImage


class ImageSubscriber(rospy.Subscriber):
    """
    Emulates image_transport for Python
    """
    def __init__(self, 
            imageTopic, 
            _callback, 
            queue_size=None,
            encoding='bgr8'):
        
        self.callTarget = _callback
        self.bridge = cv_bridge.CvBridge()
        self.encoding = encoding
        rospy.Subscriber.__init__(self, imageTopic, rospy.AnyMsg, self.callback, queue_size=queue_size)
    
    def callback(self, msg):
        connection_header = msg._connection_header['type'].split('/')
        ros_pkg = connection_header[0] + '.msg'
        msg_type = connection_header[1]
        
        if msg_type!="Image" and msg_type!="CompressedImage":
            raise ValueError("Incoming message is not of Image type")
        
        msg_class = getattr(import_module(ros_pkg), msg_type)
#         self.initsub.unregister()
#         self.deserialized_sub = rospy.Subscriber(self.topic, msg_class, self.nextMessageCallback)
        msgf = msg_class()
        realMsg = msgf.deserialize(msg._buff)

        if msg_type=='Image':
            img = self.bridge.imgmsg_to_cv2(realMsg, self.encoding)
        elif msg_type=='CompressedImage':
            img = self.bridge.compressed_imgmsg_to_cv2(realMsg, self.encoding)
        self.callTarget(img)
