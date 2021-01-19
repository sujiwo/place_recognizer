#!/usr/bin/python

"""
This script is used to perform visual training to create a topological map. 
Data source comes from a ROS Bag
"""

from RandomAccessBag import RandomAccessBag, ImageBag
import cv2
import sys
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentError
from copy import copy
from place_recognizer import VisualDictionary, VLAD2, IncrementalBoW, GeographicTrajectory, GenericTrainer
from place_recognizer.GenericImageMap import getEnhancementMethods


class BagTrainer(GenericTrainer):
    def __init__(self, method, mapfile_output, bagfilePath, imageTopic, mapfile_load=None, vdictionaryPath=None, desample=5.0, bagStart=0, bagStop=-1, enhanceMethods=False):
        super(BagTrainer, self).__init__(method, mapfile_output, mapfile_load, vdictionaryPath, useEnhancement=enhanceMethods)

        # Inspect the bag file
        self.trainBag = None
        allBagConns = RandomAccessBag.getAllConnections(bagfilePath)
        trajectorySrc = None
        for bg in allBagConns:
            if bg.topic()==imageTopic:
                self.trainBag = ImageBag(bagfilePath, imageTopic)
                print("Using {} as image source".format(self.trainBag.topic()))
            elif bg.type() in GeographicTrajectory.supportedMsgTypes:
                self.trajectorySrc = GeographicTrajectory(bg)
                print("Using {} as trajectory source".format(bg.topic()))
        if (not self.trainBag):
            raise ArgumentError("Image topic is invalid")
                
        self.sampleList = self.trainBag.desample(desample, True, bagStart, bagStop)
                
    def runTraining(self):
        # Not taking all images
        
        timestamps = [self.trainBag.messageTime(s) for s in self.sampleList]
        trajectorySrc = self.trajectorySrc.buildFromTimestamps(timestamps)

        self.initTrain()
        print("Training begin")
        i = 0
        for s in tqdm(self.sampleList):
            # Need smaller size
            curPosition = trajectorySrc.coordinates[i]
            image = self.trainBag[s]
            self.addImage(image, curPosition)
            i += 1
        
        self.stopTrain()
        print("Training Done")    



def main():
    _parser = ArgumentParser(description="VLAD & IBoW Mapping with input from ROS Bag File")
    _parser.add_argument("bagfile", type=str, metavar="image_bag")
    _parser.add_argument("topic", type=str)
    _parser.add_argument("output", type=str)
    _parser.add_argument("--dictionary", type=str, metavar="path", help="Path to initial visual dictionary (only for vlad)")
    _parser.add_argument("--method", type=str, choices=['vlad', 'ibow'], default='vlad', help="Choices of method for training")
    _parser.add_argument("--desample", type=float, metavar="hz", default=5.0, help="Reduce sample rate of images in bag file")
    _parser.add_argument("--resize", type=float, metavar="ratio", default=0.53333, help="Rescale image size with this ratio")
    _parser.add_argument("--load", type=str, metavar="path", default="", help="Load previous map for retrain")
    _parser.add_argument("--start", type=float, metavar="second", default=0, help="Start mapping from offset")
    _parser.add_argument("--mask", type=str, default='', help='Image mask file')
    _parser.add_argument("--stop", type=float, metavar="second", default=-1, help="Stop mapping at offset from 0")
    
    enhanceMethods = getEnhancementMethods()
    if len(enhanceMethods)!=0:
        _parser.add_argument("--ime", type=str, choices=enhanceMethods, help='Preprocess image with this method')
    else:
        print("Enhancement not available; install im_enhance if you want")
    
    prog_arguments = _parser.parse_args()
    
    if hasattr(prog_arguments, 'ime') and (prog_arguments.ime is not None):
        from place_recognizer.GenericImageMap import ime
        imeMethod = eval('ime.' + prog_arguments.ime)
    else:
        imeMethod = False
        
    trainer = BagTrainer(prog_arguments.method, 
                         prog_arguments.output, 
                         prog_arguments.bagfile, 
                         prog_arguments.topic, 
                         prog_arguments.load, 
                         prog_arguments.dictionary, 
                         prog_arguments.desample, 
                         prog_arguments.start, 
                         prog_arguments.stop,
                         enhanceMethods=imeMethod)
    trainer.resize_factor = prog_arguments.resize
    if prog_arguments.mask!='':
        trainer.initialMask = cv2.imread(prog_arguments.mask, cv2.IMREAD_GRAYSCALE)
        trainer.initialMask = cv2.resize(trainer.initialMask, (0,0), None, fx=trainer.resize_factor, fy=trainer.resize_factor)
        trainer.mask = copy(trainer.initialMask)
        print("Mask size now: {}".format(trainer.mask.shape))
        
    
    trainer.runTraining()
    
    


if __name__ == "__main__":
    main()
