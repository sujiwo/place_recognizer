#!/usr/bin/python

import os
import cv2
import numpy as np
from time import time


_hasSegment = False

try:
    os.environ['GLOG_minloglevel'] = '2'
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
_setSegments = np.zeros((256,), dtype=np.uint8)
_setSegments[0:defaultSegmentMask.shape[0]] = defaultSegmentMask


def RunSegment(frame, raw=False):
    origin_shape = frame.shape
    frame = cv2.resize(frame, (netInputShape[3], netInputShape[2]))
    input_image = frame.transpose((2,0,1))
    input_image = np.asarray([input_image])

    start = time()
    outp = classifier.forward_all(data=input_image)
    end = time()

    segmentation_ind = np.squeeze(outp['argmax'])
    segmentation_ind_3ch = np.resize(segmentation_ind, (3, netInputShape[2], netInputShape[3]))
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
    
    if (raw==True):
#     segmentation_rgb = cv2.LUT(segmentation_ind_3ch, label_colors)
        segmentation_bin = cv2.resize(segmentation_ind_3ch, (origin_shape[1], origin_shape[0]), None, interpolation=cv2.INTER_NEAREST)
        return segmentation_bin[:,:,0]
    else:
        segmentation_rgb = cv2.LUT(segmentation_ind_3ch, label_colors)
        return cv2.resize(segmentation_rgb, (origin_shape[1], origin_shape[0]), None, interpolation=cv2.INTER_NEAREST)

    
def CreateMask(image):
    classified = RunSegment(image, raw=True)
    return cv2.LUT(classified, _setSegments)
    
    
if __name__=="__main__" and _hasSegment==True:
    print(caffe.__version__)
    print("Caffe available")
