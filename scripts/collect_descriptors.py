#!/usr/bin/python

from RandomAccessBag import ImageBag
import cv2
from tqdm import tqdm
import numpy as np

if __name__=='__main__':
    
    orb = cv2.ORB_create(6000)
    trainBag = ImageBag('/Data/MapServer/Logs/vls128-conv.bag', '/front_rgb/image_raw')
    sampleList = trainBag.desample(5.0, True)
    descriptors = np.zeros((0,32),dtype=np.uint8)
    for s in tqdm(sampleList):
        img = trainBag[s]
        img = cv2.resize(img, (1024,576))
        k,d = orb.detectAndCompute(img, None)
        descriptors = np.append(descriptors, d, axis=0)
    
    np.save("/home/sujiwo/VmmlWorkspace/vls128-descriptors-total.npy", descriptors)