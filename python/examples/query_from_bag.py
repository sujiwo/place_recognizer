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
from place_recognizer import GenericImageDatabase
from place_recognizer.GenericImageMap import getEnhancementMethods


resizeFactor = 0.533333333


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
    samples = queryBag.desample(hz=-1, True, cmdArgs.start, cmdArgs.stop)
    positions = pool4.map(processQuery, samples)
    mlc.save(positions, cmdArgs.output)
    print("Done")
