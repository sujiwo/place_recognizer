#!/usr/bin/python

import os
import cv2
import numpy as np
from time import time


_hasSegment = False

try:
    import caffe
    _hasSegment = True
    caffe.set_mode_gpu()
except ImportError:
    print("Unable to import SegNet")
    
if _hasSegment==True:
    segnetDir = os.path.realpath(os.path.dirname(caffe.__file__)+"/../..")
    modelPath = segnetDir + "/segnet_model_driving_webdemo.prototxt"
    weightPath = segnetDir + "/segnet_weights_driving_webdemo.caffemodel"
    classifier = caffe.Net(modelPath, weightPath, caffe.TEST)
    netInputShape = classifier.blobs["data"].data.shape
    netOutputShape = classifier.blobs["argmax"].data.shape
    label_colors = cv2.imread(segnetDir + "/camvid12.png").astype(np.uint8)

label_text={
    0:'Sky',
    1:'Building',
    2:'Pole',
    3:'Road Marking',
    4:'Road',
    5:'Pavement',
    6:'Tree',
    7:'Sign Symbol',
    8:'Fence',
    9:'Vehicle',
    10:'Pedestrian',
    11:'Bike'
}

defaultSegmentMask = np.array([
    0,
    0xff,
    0xff,
    0xff,
    0xff,
    0xff,
    0,
    0xff,
    0xff,
    0,
    0,
    0
    ], dtype=np.uint8)


def RunSegment(frame, raw=False):
    origin_shape = frame.shape
    frame = cv2.resize(frame, (netInputShape[3], netInputShape[2]))
    input_image = frame.transpose((2,0,1))
    input_image = np.asarray([input_image])

    start = time()
    outp = classifier.forward_all(data=input_image)
    end = time()
    print '%30s' % 'SegNet executed in ', str((end-start)*1000), 'ms'

    if (raw==True):
        return outp['argmax'][0,0]

    segmentation_ind = np.squeeze(outp['argmax'])
    segmentation_ind_3ch = np.resize(segmentation_ind, (3, netInputShape[2], netInputShape[3]))
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
    segmentation_rgb = cv2.LUT(segmentation_ind_3ch, label_colors)
    segmentation_rgb = cv2.resize(segmentation_rgb, (origin_shape[1], origin_shape[0]), None, interpolation=cv2.INTER_NEAREST)
    return segmentation_rgb

    
def CreateMask(image):
    classified = RunSegment(image)
    
    
if __name__=="__main__" and _hasSegment==True:
    print(caffe.__version__)
    print("Caffe available")
