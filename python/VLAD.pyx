# import cv2
# Check 
# https://github.com/cmarshall108/cython-cmake-example.git
# on how to use CMake with Cython

cdef class VisualDictionaryBinary():
    
    cpdef __init__ (self, numWords=256, numFeaturesOnImage=3000):
        pass


cdef class bVLAD():
    
    cpdef __init__ (self, numWords=256, numFeaturesOnImage=3000):
        pass