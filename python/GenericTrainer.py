#!/usr/bin/env python
import sys
import cv2
from .VLAD import *
from ._place_recognizer import *


class GenericTrainer:
    """
    Base Class for training/building map file for IncrementalBoW and VLAD
    
    Attributes
    ----------
    numFeatures: number of features to be extracted by ORB descriptor \n
    mask: Image mask to be used by ORB descriptor \n
    resize_factor: Resize image frame by this factor \n
    show_image_frame: whether to show incoming image frame after preprocessed \n 
    
    Parameters
    ----------
    method: str, "ibow" or "vlad"
    mapfile_output: str, path to resulting map file
    mapfile_load: str, path to existing map file to be loaded
    vdictionaryPath: str, path to visual dictionary (only for vlad)
    """

    # Number of features to be extracted from single image
    numFeatures = 6000
    # Initial mask for feature extraction
    mask = None
    # Reduce image size with this factor
    resize_factor = 0.53333
    # Whether to show image frames
    show_image_frame = True
    
    def __init__(self, method, mapfile_output, mapfile_load=None, vdictionaryPath=None):
        """
        Initialization
        
        """
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
                # XXX: IBoW does not (yet) support saving/loading from file descriptor
                self.mapper.load(fd)
            fd.close()
        else:
            if method=="vlad":
                vdict = VisualDictionary.load(vdictionaryPath)
                self.mapper = VLAD2(vdict)
            elif method=="ibow":
                self.mapper = IncrementalBoW()
        self.extractor = cv2.ORB_create(self.numFeatures)
        
        # Image ID
        self.imageIdNext = self.mapper.lastImageId()
    
    def preprocess(self, image):
        """
        Preprocess input image frame prior to train. You may need to override this function
        in order to customize training process; eg. add segmentation or enhance the contrast
        """
        imgprep = cv2.resize(image, (0,0), None, fx=self.resize_factor, fy=self.resize_factor)
        return imgprep
    
    def addImage(self, image, imageMetadata=None):
        image_prep = self.preprocess(image)
        keypoints, descriptors = self.extractor.detectAndCompute(image_prep, self.mask)
        
        self.mapper.addImage(descriptors, keypoints, self.imageIdNext)
        self.imageIdNext += 1

        if self.show_image_frame:
            cv2.imshow("Image", image_prep)

    def initTraining(self):
        """
        Initialize a new training session
        """
        pass
    
    def stopTraining(self):
        """
        Stop current training session and save resulting map to disk
        """
        fd = open(self.mapfile_output, "wb")
        self.mapper.save(fd)
        fd.close()

