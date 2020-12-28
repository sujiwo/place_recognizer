#!/usr/bin/python

"""
This script is used to perform visual training to create a topological map. 
Data source comes from a ROS Bag
"""

from RandomAccessBag import RandomAccessBag, ImageBag
from multiprocessing import Lock
import cv2
import sys
from place_recognizer import VisualDictionary, VLAD2, IncrementalBoW, GeographicTrajectory
from tqdm import tqdm
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="VLAD & IBoW Mapping with input from ROS Bag File")
    parser.add_argument("bagfile", type=str, metavar="image_bag")
    parser.add_argument("topic", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--dictionary", type=str, metavar="path", help="Path to initial visual dictionary (only for vlad)")
    parser.add_argument("--method", type=str, choices=['vlad', 'ibow'], default='vlad', help="Choices of method for training")
    parser.add_argument("--desample", type=float, metavar="hz", default=5.0, help="Reduce sample rate of images in bag file")
    parser.add_argument("--resize", type=float, metavar="ratio", default=0.53333, help="Rescale image size with this ratio")
    parser.add_argument("--load", type=str, metavar="path", default="", help="Load previous map for retrain")
    parser.add_argument("--start", type=float, metavar="second", default=0, help="Start mapping from offset")
    parser.add_argument("--stop", type=float, metavar="second", default=-1, help="Stop mapping at offset from 0")
    cmdArgs = parser.parse_args()
    
    trainBag = ImageBag(cmdArgs.bagfile, cmdArgs.topic)
    orb = cv2.ORB_create(6000)

    mapper = None

    if (cmdArgs.method=="vlad"):
        if (cmdArgs.dictionary is None):
            print("Dictionary for vlad must be specified")
            exit(-1)
        visdict = VisualDictionary.load(cmdArgs.dictionary)
        mapper = VLAD2(visdict)
    elif (cmdArgs.method=="ibow"):
        mapper = IncrementalBoW()
    mapper.initTrain()
    
    # Not taking all images
    sampleList = trainBag.desample(cmdArgs.desample, True, cmdArgs.start, cmdArgs.stop)
    
    for s in tqdm(sampleList):
        # Need smaller size
        img = cv2.resize(trainBag[s], (0,0), None, fx=cmdArgs.resize, fy=cmdArgs.resize)
        # XXX: should have taken Segmentation results here
        k, d = orb.detectAndCompute(img, None)
        mapper.addImage(s, d, k)
        cv2.imshow("Current Image", img)
        cv2.waitKey(1)
    
    mapper.stopTrain()
    mapper.save(cmdArgs.output)
    print("Done")    


if __name__ == "__main__":
    main()
