#!/usr/bin/python

"""
Python modules that contain class to process Oxford dataset like a sequence
"""
from os import path
import sys
import cv2
import numpy as np
import csv
import rospy
from .GeographicCoordinate import GeographicTrajectory


class OxfordDataset:

    raw = False
    scale = 1.0
    
    def __init__(self, datasetDir):
        self.datasetPath = path.abspath(datasetDir)
        self.timestamps = np.loadtxt(path.join(self.datasetPath, 'stereo.timestamps'), dtype=np.int)
        self.timestamps = self.timestamps[0:self.timestamps.shape[0]-1,0]
        self._loadGps()
        
    def _loadGps(self):
        gpsPath = path.join(self.datasetPath, 'gps', 'gps.csv')
        self.gps = GeographicTrajectory()
        with open(gpsPath, 'r') as fd:
            csvfd = csv.DictReader(fd, delimiter=',')
            for r in csvfd:
                self.gps.timestamps.append(rospy.Time( nsecs=int(r['timestamp'])*1000 ))
                self.gps.coordinates.append([ float(r['easting']), float(r['northing']), float(r['altitude']) ])
        self.gps.coordinates = np.array(self.gps.coordinates)
    
    def __len__(self):
        return len(self.timestamps)
    
    def loadGroundTruth(self, gtpath):
        pass
    
    def loadDistortionModel(self, distModelPath):
        pass
    
    def __getitem__(self, i):
        imagePath = path.join(self.datasetPath, 'stereo', 'centre', str(self.timestamps[i])+'.png')
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        if self.raw==True:
            return img
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2RGB)
            img = cv2.resize(img, (0,0), None, fx=self.scale, fy=self.scale)
            return img
        


