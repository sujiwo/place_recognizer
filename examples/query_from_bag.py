#!/usr/bin/python

"""
This script is used to perform query against VLAD map in parallel

Usage:
$ query_from_bag.py <path_to_image_bag> <image_topic> <path_to_map_file> <training_output>
"""

import mlcrate as mlc
from RandomAccessBag import RandomAccessBag, ImageBag
from multiprocessing import Lock
from geodesy import utm
import cv2
import sys
from place_recognizer import VisualDictionary, VLAD2

queryBag = ImageBag(sys.argv[1], sys.argv[2])
mapvlad = VLAD2.load(sys.argv[3])
bagLock = Lock()
orb = cv2.ORB_create(6000)

def processQuery(i):
    # Bag class is not thread-safe
    bagLock.acquire()
    img=queryBag[i]
    bagLock.release()
    
    img = cv2.resize(img, (1024,576))
    k,d = orb.detectAndCompute(img, None)
    return mapvlad.query(d, numOfImages=50)

if __name__=="__main__":
    print("Ready")
    pool4 = mlc.SuperPool(n_cpu=4)
    positions = pool4.map(processQuery, range(len(queryBag)))
    mlc.save(positions, "queryResults.dat")
    print("Done")
