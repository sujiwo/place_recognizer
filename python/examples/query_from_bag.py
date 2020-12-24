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
from place_recognizer import IncrementalBoW, VisualDictionary, VLAD2, VLADLoadError


resizeFactor = 0.533333333


def processQuery(i):
    # Bag class is not thread-safe
    bagLock.acquire()
    img=queryBag[i]
    bagLock.release()
    
    img = cv2.resize(img, (0,0), None, fx=resizeFactor, fy=resizeFactor)
    k,d = orb.detectAndCompute(img, None)
    return mapsource.query(d, numOfImages=50)

if __name__=="__main__":
    
    parser = ArgumentParser(description="VLAD/IBoW Query against a ROS Bag File")
    parser.add_argument("bagfile", type=str, metavar="image_bag")
    parser.add_argument("topic", type=str)
    parser.add_argument("mapfile", type=str)
    parser.add_argument("output", type=str)
    cmdArgs = parser.parse_args()

    queryBag = ImageBag(cmdArgs.bagfile, cmdArgs.topic)
    mapsource = None
    
    try:
        mapsource = VLAD2.load(cmdArgs.mapfile)
        print("Map is of VLAD type")
    except VLADLoadError:
        mapsource = IncrementalBoW()
        mapsource.load(cmdArgs.mapfile)
        print("Map is of IBoW type")
    except Exception:
        print("Unable to determine map type")
        exit(-1)
    
    bagLock = Lock()
    orb = cv2.ORB_create(6000)
    
    print("Ready")
    pool4 = mlc.SuperPool(n_cpu=4)
    positions = pool4.map(processQuery, range(len(queryBag)))
    mlc.save(positions, cmdArgs.output)
    print("Done")
