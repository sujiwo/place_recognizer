#!/usr/bin/python

"""
This script is used to perform training against VLAD map in parallel
Data source comes from a ROS Bag

Usage:
$ train_from_bag.py <path_to_image_bag> <image_topic> <path_to_cityscape_visual_dictionary> <training_output>
"""

from RandomAccessBag import RandomAccessBag, ImageBag
from multiprocessing import Lock
import cv2
import sys
from place_recognizer import VisualDictionary, VLAD2
from tqdm import tqdm
from argparse import ArgumentParser

if __name__=="__main__":

    parser = ArgumentParser(description="VLAD Training with input from ROS Bag File")
    parser.add_argument("bagfile", type=str, metavar="image_bag")
    parser.add_argument("topic", type=str)
    parser.add_argument("visual_dictionary", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--desample", type=float, metavar="hz", default=5.0, help="Reduce sample rate of images in bag file")
    parser.add_argument("--resize", type=float, metavar="ratio", default=0.53333, help="Rescale image size with this ratio")
    cmdArgs = parser.parse_args()
    
    trainBag = ImageBag(cmdArgs.bagfile, cmdArgs.topic)
    orb = cv2.ORB_create(6000)

    visdict = VisualDictionary.load(cmdArgs.visual_dictionary)
    mapvlad = VLAD2(visdict)
    mapvlad.initTrain()
    
    # Not taking all images
    sampleList = trainBag.desample(cmdArgs.desample, True)
    
    for s in tqdm(sampleList):
        # Need smaller size
        img = cv2.resize(trainBag[s], (0,0), None, fx=cmdArgs.resize, fy=cmdArgs.resize)
        # XXX: should have taken Segmentation results here
        k, d = orb.detectAndCompute(img, None)
        mapvlad.addImage(s, d)
        cv2.imshow("Current Image", img)
        cv2.waitKey(1)
    
    mapvlad.stopTrain()
    mapvlad.save(cmdArgs.output)
    print("Done")