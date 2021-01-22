#!/usr/bin/python

"""
This script is used to perform query against VLAD/iBoW map in parallel

Usage:
$ query_from_bag.py <path_to_image_bag> <image_topic> <path_to_map_file> <query_output>
"""

import mlcrate as mlc
from RandomAccessBag import RandomAccessBag, ImageBag
from multiprocessing import Lock
from geodesy import utm
import cv2
import sys
from argparse import ArgumentParser
from place_recognizer import GenericImageDatabase, GeographicTrajectory
from place_recognizer.GenericImageMap import getEnhancementMethods
from train_from_bag import BagTrainer


resizeFactor = 0.533333333


class QueryFromBag (object):
    """
    Class to run query from a bag file and collect results in Python pickled file
    
    Parameters
    ----------
    - imageBag
    - trajectory
    - startOffset: start query from this second (0=start of bag)
    - stopOffset: stop query at this second
    - numToReturn:
    """
    startOffset = 0
    stopOffset = -1
    numToReturn = 10
    numCPU = 1
    output_path = None
    
    def __init__ (self, bag_path, map_file_path, image_topic=None, enhanceMethod=None):
        self.imageBag, self.trajectoryBag = BagTrainer.probeBagForImageAndTrajectory(bag_path, image_topic)
        self.engine = GenericImageDatabase(map_file_path)
        
        if callable(enhanceMethod):
            self.engine.useEnhancement = True
            self.engine.enhanceMethod = enhanceMethod
        self.bagLock = Lock()

    def processQuery(self, i):
        self.bagLock.acquire()
        image = self.imageBag[i]
        self.bagLock.release()
        return self.engine.query(image, numOfImages=self.numToReturn)
    
    def createQueryTrajectory(self):
        pass

    def runQuery(self):
        self.queryTrajectory = GeographicTrajectory(self.trajectoryBag)
        # XXX: query trajectory should be made on specified time range
        
        samples = self.imageBag.desample(-1, True, startOffsetTime=self.startOffset, stopOffsetTime=self.stopOffset)
        print("Ready")
        pool4 = mlc.SuperPool(n_cpu=self.numCPU)
        positions = pool4.map(self.processQuery, samples)
        print("Done")
        
        return positions


def processQuery(i):
    # Bag class is not thread-safe
    bagLock.acquire()
    img=queryBag[i]
    bagLock.release()
    
    return mapsource.query(d, numOfImages=50)

if __name__=="__main__":
    
    parser = ArgumentParser(description="VLAD/IBoW Query against a ROS Bag File")
    parser.add_argument("bagfile", type=str, metavar="image_bag")
    parser.add_argument("topic", type=str)
    parser.add_argument("mapfile", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--start", type=float, metavar="second", default=0, help="Start query from offset")
    parser.add_argument("--stop", type=float, metavar="second", default=-1, help="Stop query at offset from 0")
    
    enhanceMethods = getEnhancementMethods()
    if len(enhanceMethods)!=0:
        parser.add_argument("--ime", type=str, choices=enhanceMethods, help='Preprocess image with this method')
    else:
        print("Enhancement not available; install im_enhance if you want")

    cmdArgs = parser.parse_args()

    queryBag = ImageBag(cmdArgs.bagfile, cmdArgs.topic)
    mapsource = None
    
    mapsource = GenericImageDatabase(cmdArgs.mapfile)
    
    if hasattr(cmdArgs, 'ime') and (cmdArgs.ime is not None):
        from place_recognizer.GenericImageMap import ime
        mapsource.useEnhancement = True
        mapsource.enhanceMethod = eval('ime.' + prog_arguments.ime) 
    else:
        imeMethod = False
    
    bagLock = Lock()
    orb = cv2.ORB_create(6000)
    
    print("Ready")
    pool4 = mlc.SuperPool(n_cpu=1)
    samples = queryBag.desample(-1, True, cmdArgs.start, cmdArgs.stop)
    positions = pool4.map(processQuery, samples)
    mlc.save(positions, cmdArgs.output)
    print("Done")
