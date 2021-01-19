#!/usr/bin/python

"""
This script is an example to build initial dictionary from Cityscape training dataset
"""
import cv2
from argparse import ArgumentParser
from glob import glob
from place_recognizer import VisualDictionary
from tqdm import tqdm
import random


class CityscapeDataset:
    def __init__(self, path_to_cityscape):
        self.path = path_to_cityscape
        self.allImageFiles = glob(path_to_cityscape + "/leftImg8bit/*/*/*png")
        self.resize = 1.0
        
    def __len__(self):
        return len(self.allImageFiles)
    
    def __getitem__(self, i):
        img = cv2.imread(self.allImageFiles[i])
        return cv2.resize(img, (0,0), None, fx=self.resize, fy=self.resize)


if __name__=="__main__":
    
    parser = ArgumentParser(description="VLAD Visual Dictionary creation from Cityscape dataset")
    parser.add_argument("source_dataset", type=str, metavar="path_to_cityscape_dataset")
    parser.add_argument("output", type=str)
    parser.add_argument("--resize", type=float, default=1.0, metavar="factor", help="Resize images with this factor")
    parser.add_argument("--random", type=int, default=-1, metavar="R", help="Only take R random samples from dataset")
    cmdArgs = parser.parse_args()

    dataset = CityscapeDataset(cmdArgs.source_dataset)
    dataset.resize = cmdArgs.resize
    visdict = VisualDictionary(numWords=256, numFeaturesOnImage=6000)
    
    print("Reading all images...")
    if cmdArgs.random!=-1:
        samples = random.sample(range(len(dataset)), cmdArgs.random)
    else:
        samples = range(len(dataset))
        
    for ip in tqdm(samples):
        img = dataset[ip]
        visdict.train(img)
        cv2.imshow('image', img)
        cv2.waitKey(1)
    visdict.build()

    visdict.save(cmdArgs.output)
    print("Done")
