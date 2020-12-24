#!/usr/bin/python

"""
This script is an example to build initial dictionary from Cityscape training dataset
"""
import cv2
from argparse import ArgumentParser
from glob import glob
from place_recognizer import VisualDictionary
from tqdm import tqdm


if __name__=="__main__":
    
    parser = ArgumentParser(description="VLAD Visual Dictionary creation from Cityscape dataset")
    parser.add_argument("source_dataset", type=str, metavar="path_to_cityscape_dataset")
    parser.add_argument("output", type=str)
    parser.add_argument("--resize", type=float, default=1.0, metavar="factor", help="Resize images with this factor")
    cmdArgs = parser.parse_args()

    allImageFiles = glob(cmdArgs.source_dataset + "/leftImg8bit/*/*/*png")
    visdict = VisualDictionary(numWords=256, numFeaturesOnImage=6000)
    
    print("Reading all images...")
    for f in tqdm(allImageFiles):
        img = cv2.imread(f)
        img = cv2.resize(img, (0,0), None, fx=cmdArgs.resize, fy=cmdArgs.resize)
        visdict.train(img)
        cv2.imshow('image', img)
        cv2.waitKey(1)
    visdict.build()

    visdict.save(cmdArgs.output)
    print("Done")
