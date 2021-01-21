#!/usr/bin/python2

import cv2
import numpy as np


def ConvertColor2GrayAdj(image_bgr, k=4, alpha=0.5):
    imageYCrCb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mean_cr = np.mean(imageYCrCb[:,:,1])
    mean_cb = np.mean(imageYCrCb[:,:,2])
    
    P = np.zeros(image_bgr.shape[0:2])
    CrCb = imageYCrCb[:,:,1] - imageYCrCb[:,:,2]
    Yc = np.multiply(k*(mean_cr-mean_cb)*(CrCb), np.abs(CrCb)**alpha)
    
    return imageYCrCb[:,:,0] + Yc