#!/usr/bin/env python
import sys
import cv2
from .VLAD import *
from ._place_recognizer import *

class GenericTrainer:
    # Number of features to be extracted from single image
    numFeatures = 6000
    # Initial mask for feature extraction
    mask = None
    # Reduce image size with this factor
    resize_factor = 0.53333
    # Whether to show image frames
    show_image_frame = True
    
    def __init__(self, method, mapfile_output, mapfile_load=None, vdictionaryPath=None):
        self.method = method
        self.mapfile_output = mapfile_output
        
        # Prepare the map
        if mapfile_load!='':
            fd = open(mapfile_load, "rb")
            if method=="vlad":
                self.mapper = VLAD2.load(fd)
                print("VLAD file loaded")
            elif method=="ibow":
                self.mapper = IncrementalBoW()
                # XXX: IBoW does not support saving/loading from file descriptor
                self.mapper.load(fd)
            fd.close()
        else:
            if method=="vlad":
                vdict = VisualDictionary.load(vdictionaryPath)
                self.mapper = VLAD2(vdict)
            elif method=="ibow":
                self.mapper = IncrementalBoW()
        self.extractor = cv2.ORB_create(self.numFeatures)
    
    def preprocess(self, image):
        imgprep = cv2.resize(image, (0,0), None, fx=self.resize_factor, fy=self.resize_factor)
        return imgprep
    
    def addImage(self, image, imageMetadata=None):
        image_prep = self.preprocess(image)
        keypoints, descriptors = self.extractor.detectAndCompute(image_prep, self.mask)
        
        if self.show_image_frame:
            cv2.imshow("Image", image_prep)
        pass
    
    def stopTraining(self):
        fd = open(self.mapfile_output, "wb")
        self.mapper.save(fd)
        fd.close()
