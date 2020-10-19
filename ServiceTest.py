#!/usr/bin/env python
# coding: utf-8

# In[10]:


from RandomAccessBag import ImageBag
from place_recognizer.srv import place_recognizer
import numpy as np
import pickle
import rospy
from cv_bridge import CvBridge

bridge = CvBridge()
def runPlaceRecognizerImg(imageArray):
    msg = bridge.cv2_to_imgmsg(imageArray, "bgr8")
    server = rospy.ServiceProxy('place_recognizer', place_recognizer)
    placeResp = server(msg)
    return placeResp.keyframeId

trainImagesLoc=np.loadtxt('/media/sujiwo/PlaceRecognition/vls128-conv.image.csv')
queryImagesLoc=np.loadtxt('/media/sujiwo/PlaceRecognition/ouster64-conv.image.csv')

queryBag=ImageBag('/media/sujiwo/PlaceRecognition/ouster64-prep-4.bag', '/front_rgb/image_raw')

distanceThreshold = 15.0

print("Test Length: {}".format(len(queryBag)))
print("Train Length: {}".format(len(trainImagesLoc)))

queryResults={}
for ti in range(len(queryBag)):
    print("Testing: {}".format(ti))
    image = queryBag[ti]
    qres=runPlaceRecognizerImg(image)
    rights=0
    wrongs=0
    for u in qres:
        loc=trainImagesLoc[u,1:4]
        dist=np.linalg.norm(loc-queryImagesLoc[ti,1:4])
        if (dist<=distanceThreshold):
            rights +=1
        else:
            wrongs +=1
    print("--- {}: {} True, {} False".format(ti, rights, wrongs))
    queryResults[ti] = [rights, wrongs]

pickle.dump(queryResults, open("/media/sujiwo/PlaceRecognition/test-prep4.pickle", "wb"))
print("Done")


# In[ ]:




