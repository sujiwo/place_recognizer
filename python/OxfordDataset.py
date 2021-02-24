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
from copy import copy
from tf import transformations as tfx
from bisect import bisect
from .GeographicCoordinate import GeographicTrajectory


class OxfordDataset:

    gps = None
    rtk = None
    raw = False
    scale = 1.0
    distortionLUT_center_x = None
    distortionLUT_center_y = None
    originCorrectionEasting = -620248.53
    originCorrectionNorthing = -5734882.47
    originCorrectionAltitude = 0
    trajectory = None

    
    def __init__(self, datasetDir):
        self.datasetPath = path.abspath(datasetDir)
        _timestamps = np.loadtxt(path.join(self.datasetPath, 'stereo.timestamps'), dtype=np.int)
        _timestamps = _timestamps[0:_timestamps.shape[0]-1,0]
        self.filetimes = copy(_timestamps)
        self.timestamps = [ rospy.Time(nsecs=t*1000) for t in _timestamps]
        self._loadGps()
        
    def _loadGps(self):
        gpsPath = path.join(self.datasetPath, 'gps', 'gps.csv')
        self.gps = GeographicTrajectory()
        with open(gpsPath, 'r') as fd:
            csvfd = csv.DictReader(fd, delimiter=',')
            for r in csvfd:
                self.gps.timestamps.append(rospy.Time( nsecs=int(r['timestamp'])*1000 ))
                self.gps.coordinates.append([ 
                    float(r['easting']), 
                    float(r['northing']), 
                    float(r['altitude']) ])
        self.gps.coordinates = np.array(self.gps.coordinates) + \
            [self.originCorrectionEasting, 
             self.originCorrectionNorthing, 
             self.originCorrectionAltitude]
    
    def __len__(self):
        return len(self.filetimes)
    
    def loadGroundTruth(self, gtpath):
        """
        Loads pre-computed ground truth derived from RTK-GPS in a directory, usually with base name 'ground_truth_rtk'
        """
        gt_csv_path = path.join(gtpath, 'rtk', path.basename(self.datasetPath), 'rtk.csv')
        self.rtk = GeographicTrajectory()
        with open(gt_csv_path, 'r') as fd:
            csvfd = csv.DictReader(fd, delimiter=',')
            for r in csvfd:
                self.rtk.timestamps.append(rospy.Time( nsecs=int(r['timestamp'])*1000 ))
                q = tfx.quaternion_from_euler(float(r['roll']), float(r['pitch']), float(r['yaw']))
                self.rtk.coordinates.append([ 
                    float(r['easting']), 
                    float(r['northing']), 
                    float(r['altitude']),
                    q[0], q[1], q[2], q[3] ])
        self.rtk.coordinates = np.array(self.rtk.coordinates) + \
            [self.originCorrectionEasting, 
             self.originCorrectionNorthing, 
             self.originCorrectionAltitude,
             0,0,0,0]
            
        # Build image pose ground truths
        self.trajectory = GeographicTrajectory()
        self.trajectory.timestamps = self.timestamps
        self.trajectory.coordinates = []
        for t in self.timestamps:
            if t < self.rtk.timestamps[0]:
                self.trajectory.coordinates.append(self.rtk.coordinates[0])
            elif t > self.rtk.timestamps[-1]:
                self.trajectory.coordinates.append(self.rtk.coordinates[-1])
            else:
                p = self.rtk.positionAt(t)
                self.trajectory.coordinates.append(p)
        self.trajectory.coordinates = np.array(self.trajectory.coordinates)
        
    
    def loadDistortionModel(self, distModelDir):
        """
        Load distorsion coefficient file.
        - distModelDir: str, path to distorsion correction directory, downloaded from Oxford Robotcar SDK website
        """
        lutfd = open(path.join(distModelDir, 'stereo_narrow_left_distortion_lut.bin'), 'rb')
        lutfd.seek(0, 2)
        lutfdsize = lutfd.tell() 
        if (lutfdsize % 8 != 0):
            raise IOError("File size is incorrect")
        lutfd.seek(0)
        self.distortionLUT_center_x = np.fromfile(lutfd, dtype=np.double, count=(lutfdsize/2)/8)
        self.distortionLUT_center_y = np.fromfile(lutfd, dtype=np.double, count=(lutfdsize/2)/8)
        self.distortionLUT_center_x = self.distortionLUT_center_x.astype(np.float32).reshape((960,1280))
        self.distortionLUT_center_y = self.distortionLUT_center_y.astype(np.float32).reshape((960,1280))
        
    def _imageFileName(self, i):
        return path.join(self.datasetPath, 'stereo', 'centre', str(self.filetimes[i])+'.png')
    
    def __getitem__(self, i):
        """
        Returns image at index position i
        """
        imagePath = self._imageFileName(i)
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        if self.raw==True:
            return img
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2RGB)
            img = cv2.resize(img, (0,0), None, fx=self.scale, fy=self.scale)
            if (self.distortionLUT_center_x is None):
                return img
            else:
                return self.undistort(img)
        
    def undistort(self, image):
        if (self.distortionLUT_center_x is None):
            raise ValueError("Distortion coefficient has not been loaded")
        return cv2.remap(image, self.distortionLUT_center_x, self.distortionLUT_center_y, cv2.INTER_LINEAR)
    
    def position(self, i):
        if (self.trajectory is None):
            raise ValueError("Error: ground truth has not been loaded")
        return self.trajectory.coordinates[i]

    def desample(self, hz):
        lengthInSeconds = (self.timestamps[-1]-self.timestamps[0]).to_sec()
        if hz >= len(self) / lengthInSeconds:
            raise ValueError("Frequency must be lower than the original one")
        
        pointer = []
        tInterval = 1.0 / float(hz)
        td = 0.0
        for twork in np.arange(td, td+lengthInSeconds, 1.0):
            tMax = min(twork+1.0, td+lengthInSeconds)
            tm = twork + tInterval
            while tm < tMax:
                curtime = self.timestamps[0] + rospy.Duration.from_sec(tm)
                if curtime == self.timestamps[0]:
                    idx = 0
                else:
                    idx = bisect(self.timestamps, curtime)
                pointer.append(idx)
                tm += tInterval
        
        return pointer

        
        
        
        
